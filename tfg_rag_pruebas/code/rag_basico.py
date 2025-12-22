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
        print("Iniciando sistema RAG (Modo: Robust Re-ranking v2 - Extended Context)...")
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
        # 1. EXTRACCIÓN
        extract_tmpl = """
        Analiza la petición del usuario. Extrae 3-5 palabras clave técnicas en inglés.
        Petición: {user_request}
        Responde SOLO con las palabras separadas por comas.
        """
        self.extraction_chain = ChatPromptTemplate.from_template(extract_tmpl) | self.llm | StrOutputParser()

        # 2. FILTRADO (Prompt más estricto)
        filter_tmpl = """
        Eres un filtro de metadatos JSON. TU ÚNICA SALIDA DEBE SER UN JSON VÁLIDO.
        NO escribas introducciones, ni explicaciones, ni "Here is the JSON". SOLO EL JSON.

        Objetivo: Analiza la lista de candidatos y selecciona los archivos que responden a la intención del usuario: "{user_request}".
        
        Reglas:
        - Intención específica (ej. "Buildings") -> Descarta otros dominios (ej. "Agriculture").
        - Intención genérica -> Sé permisivo.
        
        Candidatos:
        {context_list}

        Formato de Salida OBLIGATORIO:
        {{
            "relevant_sources": ["archivo1.ttl", "archivo2.owl"]
        }}
        """
        self.filter_chain = ChatPromptTemplate.from_template(filter_tmpl) | self.llm | StrOutputParser()

        # 3. DECISIÓN FINAL (Prohibiendo URIs)
        selection_tmpl = """
        Eres un Experto en Ontologías. Recomienda el MEJOR archivo de la lista.
        
        Petición: {user_request}
        Contexto (Candidatos Filtrados):
        {filtered_context}
        
        INSTRUCCIÓN CRÍTICA:
        En el campo "ONTOLOGÍA RECOMENDADA", debes poner EXACTAMENTE el nombre del archivo (el texto que aparece después de 'Fuente:').
        PROHIBIDO poner URIs (http://...) o nombres de clases. SOLO el nombre del archivo (ej. bot.ttl).
        
        Respuesta:
        **ONTOLOGÍA RECOMENDADA:** [NOMBRE_DEL_ARCHIVO]
        **RAZÓN:** [Justificación breve]
        """
        self.selection_chain = ChatPromptTemplate.from_template(selection_tmpl) | self.llm | StrOutputParser()

    def _extract_json_from_text(self, text):
        """Función auxiliar para limpiar la basura conversacional del LLM"""
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
            return json.loads(text)
        except:
            return None

    def run_pipeline(self, user_request, initial_k=40):
        start_time = time.time()
        
        # 1. Extracción
        try:
            keywords = self.extraction_chain.invoke({"user_request": user_request})
        except: keywords = user_request

        # 2. Broad Retrieval
        raw_docs = self.vectorstore.max_marginal_relevance_search(keywords, k=initial_k, fetch_k=100)
        
        # Preparar contexto para filtro
        doc_summaries = []
        for d in raw_docs:
            src = d.metadata.get('source', 'unknown')
            # AUMENTADO A 450 CARACTERES PARA LEER DESCRIPCIONES COMPLETAS
            content = d.page_content[:450].replace('\n', ' ')
            doc_summaries.append(f"- FILE: {src} | CONTENT: {content}...")
        doc_list_str = "\n".join(doc_summaries)
        
        # 3. Re-ranking con Regex Parsing
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
                relevant_files = [d.metadata.get('source') for d in raw_docs[:5]]
                
        except Exception as e:
            print(f"⚠️ Fallback filtro: {e}")
            relevant_files = [d.metadata.get('source') for d in raw_docs[:5]]

        if not isinstance(relevant_files, list): relevant_files = []

        # 4. Reconstrucción Contexto
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