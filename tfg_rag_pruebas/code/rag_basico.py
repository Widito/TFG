import os
import time
import json
import re
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
        print("Iniciando sistema RAG (Modo: Agnóstico y Multilingüe)...")
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
        # 1. EXTRACCIÓN (Flexible y Multilingüe)
        extract_tmpl = """
        Analiza la petición del usuario para buscar ontologías.
        Petición: {user_request}
        
        Tarea: Extrae las palabras clave técnicas o conceptos principales necesarios para realizar la búsqueda.
        - NO hay límite de palabras: extrae todas las necesarias.
        - Mantén el idioma original si es relevante, o usa terminología técnica estándar.
        
        Responde SOLO con las palabras clave separadas por comas.
        """
        self.extraction_chain = ChatPromptTemplate.from_template(extract_tmpl) | self.llm | StrOutputParser()

        # 2. FILTRADO (LÓGICA PURA, SIN SESGOS DE NOMBRES)
        filter_tmpl = """
        Eres un filtro de relevancia semántica para un motor de búsqueda de ontologías.
        
        INPUT:
        - Query del usuario: "{user_request}"
        - Lista de fragmentos recuperados (Candidatos).

        TAREA:
        Identifica qué archivos son RELEVANTES para responder a la query basándote en su CONTENIDO.

        CRITERIOS DE SELECCIÓN (LÓGICOS):
        1. **Coincidencia Conceptual:** Si un fragmento define una Clase o Propiedad que se busca (ej. Query: "Sensor", Fragmento: "Class: Sensor"), es relevante.
        2. **Cobertura Parcial:** Si la query es compleja (ej. "Zonas y Sensores"), mantén cualquier archivo que cubra AL MENOS UNO de los conceptos. NO descartes un archivo porque le falte una parte; otro archivo puede completarla.
        3. **Dominio:** Descarta archivos cuyo dominio sea claramente incompatible (ej. si buscan "biología", descarta "finanzas"). Si hay duda, MANTÉN el archivo.

        Candidatos:
        {context_list}

        FORMATO DE RESPUESTA (JSON PURO):
        {{
            "relevant_sources": ["archivo1.ttl", "archivo2.n3"]
        }}
        """
        self.filter_chain = ChatPromptTemplate.from_template(filter_tmpl) | self.llm | StrOutputParser()

        # 3. DECISIÓN FINAL (Neutral)
        selection_tmpl = """
        Eres un Experto en Web Semántica. Selecciona la ontología más adecuada de la lista proporcionada.
        
        Petición: {user_request}
        Candidatos Filtrados:
        {filtered_context}
        
        Instrucciones:
        1. Analiza qué ontología define mejor los conceptos centrales de la petición.
        2. Prioriza la especialización: Si una ontología parece diseñada específicamente para el problema del usuario, elígela sobre una genérica.
        3. Justifica tu decisión basándote ÚNICAMENTE en la evidencia visible en los fragmentos (Clases, Propiedades, Descripciones).
        
        Respuesta OBLIGATORIA:
        **ONTOLOGÍA RECOMENDADA:** [NOMBRE_EXACTO_DEL_ARCHIVO]
        **RAZÓN:** [Justificación técnica breve]
        """
        self.selection_chain = ChatPromptTemplate.from_template(selection_tmpl) | self.llm | StrOutputParser()

    def _extract_json_from_text(self, text):
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return json.loads(text)
        except:
            return None

    def run_pipeline(self, user_request, initial_k=40):
        start_time = time.time()
        
        # 1. Extracción
        try: keywords = self.extraction_chain.invoke({"user_request": user_request})
        except: keywords = user_request

        # 2. Broad Retrieval
        raw_docs = self.vectorstore.max_marginal_relevance_search(keywords, k=initial_k, fetch_k=100)
        
        # Preparar contexto (Mantenemos 450 chars para ver descripciones completas)
        doc_summaries = []
        for d in raw_docs:
            src = d.metadata.get('source', 'unknown')
            content = d.page_content[:450].replace('\n', ' ')
            doc_summaries.append(f"- FILE: {src} | CONTENT: {content}...")
        doc_list_str = "\n".join(doc_summaries)
        
        # 3. Re-ranking Semántico
        relevant_files = []
        try:
            raw_filter_output = self.filter_chain.invoke({
                "user_request": user_request,
                "context_list": doc_list_str
            })
            parsed_json = self._extract_json_from_text(raw_filter_output)
            
            if parsed_json and "relevant_sources" in parsed_json:
                relevant_files = parsed_json["relevant_sources"]
            else:
                # Fallback neutro: Top 5 de Chroma
                relevant_files = [d.metadata.get('source') for d in raw_docs[:5]]
        except Exception as e:
            print(f"⚠️ Fallback filtro: {e}")
            relevant_files = [d.metadata.get('source') for d in raw_docs[:5]]

        if not isinstance(relevant_files, list): relevant_files = []

        # 4. Contexto Final
        final_docs = [d for d in raw_docs if d.metadata.get('source') in relevant_files]
        if not final_docs: final_docs = raw_docs[:3]

        context_lines = [f"- [Fuente: {d.metadata.get('source')}] {d.page_content[:450]}..." for d in final_docs]
        context_str = "\n".join(context_lines)

        # 5. Generación
        decision_text = self.selection_chain.invoke({
            "user_request": user_request,
            "filtered_context": context_str
        })

        return {
            "query": user_request,
            "keywords": keywords,
            "unique_retrieved_sources": list(set([d.metadata.get('source') for d in final_docs])),
            "llm_response": decision_text,
            "execution_time": time.time() - start_time
        }

if __name__ == "__main__":
    rag = OntologyRecommender()
    while True:
        q = input("\nConsulta ('salir'): ")
        if q == 'salir': break
        res = rag.run_pipeline(q)
        print(f"Filtrados: {res['unique_retrieved_sources']}")
        print(f"Respuesta:\n{res['llm_response']}")