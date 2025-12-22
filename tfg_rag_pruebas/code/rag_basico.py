import os
import time
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

# Configuración global
PERSIST_DIRECTORY = "tfg_rag_pruebas/chroma_db"
EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL = "llama3"

class OntologyRecommender:
    def __init__(self):
        print("Iniciando sistema RAG (Modo: Broad Retrieval + Re-ranking Agnóstico)...")
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
        # 1. CHAIN DE EXTRACCIÓN (Keywords)
        extract_tmpl = """
        Analiza la petición del usuario para buscar ontologías.
        Petición: {user_request}
        Extrae 3-5 palabras clave o conceptos técnicos principales (en inglés si es posible).
        Responde SOLO con las palabras separadas por comas.
        """
        self.extraction_chain = ChatPromptTemplate.from_template(extract_tmpl) | self.llm | StrOutputParser()

        # 2. CHAIN DE FILTRADO/RE-RANKING (El Juez Imparcial)
        # Lógica semántica que se adapta a la intención del usuario
        filter_tmpl = """
        Tienes una petición de usuario y una lista de fragmentos de ontologías candidatas recuperadas de una base de datos.
        Tu trabajo es actuar como un FILTRO INTELIGENTE para eliminar el "ruido" semántico.

        Petición del Usuario: "{user_request}"

        Instrucciones de Filtrado:
        1. Analiza la INTENCIÓN del usuario (¿Busca construcción? ¿Industria? ¿Sensores? ¿Gobierno?).
        2. Revisa la lista de candidatos.
        3. SI la petición es específica (ej. "Edificios"), DESCARTA candidatos de otros dominios (ej. "Agricultura", "Manufactura") aunque usen palabras similares.
        4. SI la petición es genérica, sé más permisivo.
        5. El objetivo es quedarse SOLO con los archivos que realmente ayudan a responder la petición.

        Lista de Candidatos:
        {context_list}

        Responde con un JSON válido que contenga la lista de nombres de archivo ("source") que SÍ son relevantes.
        Formato JSON:
        {{
            "relevant_sources": ["archivo_valido_1.ttl", "archivo_valido_2.owl"]
        }}
        """
        self.filter_chain = ChatPromptTemplate.from_template(filter_tmpl) | self.llm | JsonOutputParser()

        # 3. CHAIN DE DECISIÓN FINAL
        selection_tmpl = """
        Eres un Experto en Web Semántica y Ontologías.
        Tu tarea es recomendar la MEJOR ontología basándote únicamente en los candidatos filtrados que se te presentan.
        
        Petición del usuario: {user_request}
        
        Candidatos Relevantes (Filtrados):
        {filtered_context}
        
        Instrucciones:
        1. Si hay varias opciones, elige la más específica y estándar para el dominio.
        2. Justifica tu respuesta técnicamente basándote en las clases/propiedades que ves en los fragmentos.
        3. Ignora cualquier conocimiento externo; usa solo el contexto.
        
        Tu respuesta debe tener este formato EXACTO:
        **ONTOLOGÍA RECOMENDADA:** [nombre_exacto_del_archivo]
        **RAZÓN:** [Justificación breve]
        """
        self.selection_chain = ChatPromptTemplate.from_template(selection_tmpl) | self.llm | StrOutputParser()

    def run_pipeline(self, user_request, initial_k=40):
        start_time = time.time()
        
        # PASO 1: Extracción de Keywords
        try:
            keywords = self.extraction_chain.invoke({"user_request": user_request})
        except Exception:
            keywords = user_request # Fallback
            
        # PASO 2: Broad Retrieval (Recuperación Amplia)
        # Aumentamos k a 40 para asegurar que la respuesta correcta entre en la red, aunque haya ruido.
        raw_docs = self.vectorstore.max_marginal_relevance_search(keywords, k=initial_k, fetch_k=100)
        
        # Preparar resumen para el Re-ranker
        doc_summaries = []
        for d in raw_docs:
            src = d.metadata.get('source', 'unknown')
            # Usamos un snippet corto para que el LLM pueda procesar los 40 docs rápido
            content_preview = d.page_content[:150].replace('\n', ' ') 
            doc_summaries.append(f"- SOURCE: {src} | CONTENT: {content_preview}...")
        
        doc_list_str = "\n".join(doc_summaries)
        
        # PASO 3: Re-ranking Semántico (El Filtro)
        relevant_files = []
        try:
            # Invocamos al LLM para que filtre
            filter_result = self.filter_chain.invoke({
                "user_request": user_request,
                "context_list": doc_list_str
            })
            
            if isinstance(filter_result, dict):
                relevant_files = filter_result.get("relevant_sources", [])
            elif isinstance(filter_result, list): # A veces los LLMs devuelven lista directa
                relevant_files = filter_result
                
        except Exception as e:
            print(f"⚠️ Error en Re-ranking (JSON malformado o fallo LLM): {e}")
            # Fallback inteligente: Si falla el filtro, usamos los top 5 originales de Chroma
            relevant_files = [d.metadata.get('source') for d in raw_docs[:5]]

        # Asegurar que relevant_files es una lista de strings
        if not isinstance(relevant_files, list):
            relevant_files = []

        # PASO 4: Reconstrucción del Contexto
        # Filtramos los documentos raw dejando solo los aprobados por el LLM
        final_docs = [d for d in raw_docs if d.metadata.get('source') in relevant_files]
        
        # SAFETY NET: Si el filtro fue demasiado agresivo y no dejó nada, 
        # volvemos a los Top 3 de la búsqueda vectorial original para no dar respuesta vacía.
        if not final_docs:
            print("⚠️ El filtro descartó todo. Usando Top 3 original como fallback.")
            final_docs = raw_docs[:3]
            
        # Preparamos el contexto final detallado para la generación
        context_lines = []
        for d in final_docs:
            src = d.metadata.get('source', 'unknown')
            snippet = d.page_content[:300].replace('\n', ' ')
            context_lines.append(f"- [Fuente: {src}] {snippet}...")
        context_str = "\n".join(context_lines)

        # PASO 5: Generación Final
        decision_text = self.selection_chain.invoke({
            "user_request": user_request,
            "filtered_context": context_str
        })

        # Retornamos estructura compatible con evaluate_rag.py
        # Nota: 'unique_retrieved_sources' ahora refleja lo que pasó el filtro (Reranked)
        return {
            "query": user_request,
            "keywords": keywords,
            "retrieved_sources": [d.metadata.get('source') for d in raw_docs], # Debug: Lo que vio Chroma
            "unique_retrieved_sources": list(set([d.metadata.get('source') for d in final_docs])), # Lo que usó el LLM
            "llm_response": decision_text,
            "execution_time": time.time() - start_time
        }

if __name__ == "__main__":
    rag = OntologyRecommender()
    while True:
        q = input("\nConsulta ('salir'): ")
        if q == 'salir': break
        res = rag.run_pipeline(q)
        print(f"Fuentes Filtradas: {res['unique_retrieved_sources']}")
        print(f"Respuesta: {res['llm_response']}")