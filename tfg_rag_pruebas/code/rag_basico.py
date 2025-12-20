import os
import time
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

# Configuración global
PERSIST_DIRECTORY = "tfg_rag_pruebas/chroma_db"
EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL = "llama3"

class OntologyRecommender:
    def __init__(self):
        print("Iniciando sistema RAG...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        if not os.path.exists(PERSIST_DIRECTORY):
            raise FileNotFoundError(f"No se encuentra la BD en {PERSIST_DIRECTORY}")
            
        self.vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY, 
            embedding_function=self.embeddings
        )
        
        self.llm = ChatOllama(model=LLM_MODEL, temperature=0.0)
        self._setup_chains()
        print("Sistema RAG listo.")

    def _setup_chains(self):
        # ETAPA 1: Extracción (Sin cambios)
        extract_tmpl = """
        Eres un experto en análisis semántico. Extrae palabras clave de búsqueda.
        Petición: {user_request}
        Responde SOLO con palabras clave separadas por comas (ej: Building, Site, Storey).
        Palabras clave:
        """
        self.extraction_chain = ChatPromptTemplate.from_template(extract_tmpl) | self.llm | StrOutputParser()

        # ETAPA 3: Decisión (REFORZADA)
        selection_tmpl = """
        Eres un sistema RAG estricto. Tu trabajo es seleccionar el documento más relevante de la lista proporcionada a continuación.
        
        ADVERTENCIAS CRÍTICAS:
        1. Basa tu respuesta ÚNICAMENTE en el contexto proporcionado. NO uses conocimiento externo ni menciones ontologías que no estén en la lista (como SUMO, DOLCE, etc.).
        2. Para el nombre de la ontología, DEBES copiar exactamente el texto que aparece después de "Fuente: " en el contexto. NO uses URIs (http://...).
        
        Petición del usuario: {user_request}
        
        Contexto (Resultados de búsqueda):
        {context_with_sources}
        
        Tu respuesta debe tener este formato EXACTO:
        **ONTOLOGÍA RECOMENDADA:** [nombre_exacto_del_archivo_fuente]
        **RAZÓN:** [explicación breve basada SOLO en el texto recuperado]
        """
        self.selection_chain = ChatPromptTemplate.from_template(selection_tmpl) | self.llm | StrOutputParser()

    def run_pipeline(self, user_request, top_k=15):
        """Ejecuta el flujo completo y devuelve un diccionario con resultados detallados."""
        start_time = time.time()
        
        # 1. Extracción
        try:
            keywords = self.extraction_chain.invoke({"user_request": user_request})
        except Exception as e:
            return {"error": f"Fallo en extracción: {str(e)}"}

        # 2. Recuperación (MMR)
        docs = self.vectorstore.max_marginal_relevance_search(keywords, k=top_k, fetch_k=50)
        
        # Procesar fuentes recuperadas
        retrieved_sources = [d.metadata.get('source', 'unknown') for d in docs]
        unique_sources = list(set(retrieved_sources))
        
        # Contexto para el LLM
        context_lines = []
        for d in docs:
            src = d.metadata.get('source', 'unknown')
            snippet = d.page_content[:200].replace('\n', ' ')
            context_lines.append(f"- [Fuente: {src}] {snippet}...")
        context_str = "\n".join(context_lines)

        # 3. Generación
        decision_text = self.selection_chain.invoke({
            "user_request": user_request,
            "context_with_sources": context_str
        })

        return {
            "query": user_request,
            "keywords": keywords,
            "retrieved_sources": retrieved_sources, # Lista completa (con duplicados)
            "unique_retrieved_sources": unique_sources, # Lista única
            "llm_response": decision_text,
            "execution_time": time.time() - start_time
        }

# --- Bloque para mantener la ejecución manual ---
if __name__ == "__main__":
    rag = OntologyRecommender()
    print("\n--- Modo Interactivo ---")
    while True:
        q = input("\n¿Qué necesitas modelar? ('salir'): ")
        if q.lower() == 'salir': break
        
        result = rag.run_pipeline(q)
        print(f"\nKeywords: {result['keywords']}")
        print(f"Fuentes encontradas: {result['unique_retrieved_sources']}")
        print(f"\nRespuesta:\n{result['llm_response']}")