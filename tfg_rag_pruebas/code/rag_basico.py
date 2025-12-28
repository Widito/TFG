import os
import time
import json
import re
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document



# CONFIGURACIÓN
# Construir ruta absoluta dinámica: carpeta 'chroma_db' hermana de la carpeta 'code'
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../tfg_rag_pruebas/code
project_root = os.path.dirname(current_dir)              # .../tfg_rag_pruebas
PERSIST_DIRECTORY = os.path.join(project_root, "chroma_db")
EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL = "llama3"

class OntologyRecommender:
    def __init__(self):
        print("Iniciando sistema RAG con Búsqueda Híbrida (Dense + Sparse)...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        if not os.path.exists(PERSIST_DIRECTORY):
            raise FileNotFoundError(f"No se encuentra la BD en {PERSIST_DIRECTORY}")
            
        # 1. Cargar Base Vectorial (Chroma)
        self.vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY, 
            embedding_function=self.embeddings
        )
        
        # 2. Inicializar Retrievers
        self._setup_retrievers()
        
        # 3. Inicializar LLM y Cadenas
        self.llm = ChatOllama(model=LLM_MODEL, temperature=0.0)
        self._setup_chains()
        print("Sistema RAG Híbrido listo.")

    def _setup_retrievers(self):
        """Configura el sistema de recuperación híbrida"""
        print(" - Construyendo índice BM25 (esto puede tardar unos segundos)...")
        
        # A. Recuperar documentos crudos de Chroma para indexar con BM25
        # NOTA: Chroma guarda los textos, necesitamos sacarlos para que BM25 los procese.
        try:
            # Obtenemos todos los documentos de la colección
            collection_data = self.vectorstore.get() 
            texts = collection_data['documents']
            metadatas = collection_data['metadatas']
            
            # Reconstruimos objetos Document para LangChain
            docs = [
                Document(page_content=t, metadata=m) 
                for t, m in zip(texts, metadatas)
            ]
        except Exception as e:
            print(f"Error cargando docs para BM25: {e}")
            docs = []

        if not docs:
            raise ValueError("La base de datos Chroma parece vacía o no se pudo leer para BM25.")

        # B. Retriever disperso (Keyword Search - BM25)
        self.bm25_retriever = BM25Retriever.from_documents(docs)
        self.bm25_retriever.k = 40  # Coincidir con el k del vectorial

        # C. Retriever denso (Semantic Search - Chroma)
        self.chroma_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 40}
        )


    def _hybrid_retrieve(self, query, k=40):
        """
        Recuperación híbrida: Dense (Chroma) + Sparse (BM25)
        Combina resultados priorizando semántica pero manteniendo keywords exactas.
        """
        # Recuperación
        dense_docs = self.chroma_retriever.invoke(query)
        sparse_docs = self.bm25_retriever.invoke(query)

        # Limitar a k resultados
        dense_docs = dense_docs[:k]
        sparse_docs = sparse_docs[:k]

        seen = set()
        combined = []

        for d in dense_docs + sparse_docs:
            uid = (d.page_content, str(d.metadata))
            if uid not in seen:
                seen.add(uid)
                combined.append(d)
            if len(combined) >= k:
                break

        return combined


    def _setup_chains(self):
        # 1. EXTRACCIÓN (Sin cambios, ya era genérica)
        extract_tmpl = """
        Analiza la petición del usuario.
        Petición: {user_request}
        
        Tarea: Genera una lista de palabras clave técnicas y conceptos principales para buscar.
        - Usa terminología técnica en inglés.
        - Incluye sinónimos si es necesario.
        
        Respuesta: Solo las palabras clave separadas por comas.
        """
        self.extraction_chain = ChatPromptTemplate.from_template(extract_tmpl) | self.llm | StrOutputParser()

        # 2. FILTRADO (Hacemos el filtro agnóstico también)
        filter_tmpl = """
        Eres un filtro de relevancia semántica para ontologías.
        
        INPUT:
        - Query: "{user_request}"
        - Candidatos: Lista de fragmentos de texto (RDF/OWL).

        TAREA:
        Identifica qué archivos contienen definiciones ontológicas (Clases, Propiedades) relacionadas con los conceptos de la query.
        
        CRITERIOS DE SELECCIÓN (AGNÓSTICOS):
        1. **Presencia de Definiciones:** Busca si el término aparece como sujeto de definición (ej: "X a owl:Class", "X a rdf:Property") y no solo como comentario.
        2. **Coincidencia Semántica:** Si la query pide "X" y el archivo define "X" o un sinónimo técnico directo, selecciónalo.
        3. **Ignora el dominio:** No importa si es medicina, construcción o leyes. Si define el término, es relevante.

        SALIDA (JSON VÁLIDO):
        {{
            "relevant_sources": ["archivo1.ext", "archivo2.ext"]
        }}
        """
        self.filter_chain = ChatPromptTemplate.from_template(filter_tmpl) | self.llm | StrOutputParser()

        # 3. DECISIÓN FINAL (TOTALMENTE GENERAL / SIN EJEMPLOS DE DOMINIO)
        selection_tmpl = """
        Actúa como un Arquitecto de Conocimiento experto en Linked Open Data.
        Tu objetivo es seleccionar la MEJOR ontología (archivo fuente) para cubrir las necesidades de modelado del usuario, sea cual sea el dominio.

        PETICIÓN DEL USUARIO: "{user_request}"

        CANDIDATOS RECUPERADOS (Fragmentos de código RDF/OWL):
        {filtered_context}

        ALGORITMO DE RAZONAMIENTO (Ejecuta estos pasos mentalmente):

        1. **Análisis de Densidad de Definición:**
           - Revisa cada candidato. ¿El archivo *define* los conceptos clave de la query (usando `owl:Class`, `rdf:Property`) o solo los *menciona* (en comentarios o rangos)?
           - Prioriza siempre el archivo que contiene la definición formal (el "Dueño" del concepto).

        2. **Evaluación de Especificidad (Principio de Parsimonia):**
           - Compara los candidatos que pasaron el paso 1.
           - Si un candidato parece ser una **Ontología Núcleo/Core** (definiciones generales, fundamentales) y otro es una **Extensión/Vertical** (importa otras, nombres muy largos o específicos de un nicho):
             -> **REGLA:** Elige la Ontología Núcleo SI la petición del usuario es general.
             -> **REGLA:** Elige la Extensión SOLO SI la petición del usuario incluye detalles específicos de ese nicho (ej: métricas específicas, restricciones de dominio).

        3. **Desambiguación por Contexto:**
           - Si dos ontologías definen el mismo término (homónimos), lee las propiedades (`rdfs:comment`, `rdfs:label`) en los fragmentos. Elige la que semánticamente se alinee mejor con la intención de la query.

        SALIDA REQUERIDA (JSON):
        {{
            "ONTOLOGÍA_RECOMENDADA": "nombre_exacto_del_archivo",
            "RAZÓN": "Justificación técnica basada en la densidad de definición y nivel de abstracción (Core vs Extensión)."
        }}
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

        # 2. Retrieval HÍBRIDO (Ensemble)
        # Recuperación híbrida manual (Dense + Sparse)
        # usa el configurado en __init__ (que pusimos a 40).
        raw_docs = self._hybrid_retrieve(keywords, k=40)
        
        # Preparar contexto
        doc_summaries = []
        for d in raw_docs:
            src = d.metadata.get('source', 'unknown')
            # Limpiamos saltos de línea para ahorrar tokens
            content = d.page_content[:450].replace('\n', ' ') 
            doc_summaries.append(f"- FILE: {src} | CONTENT: {content}...")
        doc_list_str = "\n".join(doc_summaries)
        
        # 3. Re-ranking (Filtrado con LLM)
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
                relevant_files = [d.metadata.get('source') for d in raw_docs[:15]]
        except Exception as e:
            print(f"Fallback filtro: {e}")
            relevant_files = [d.metadata.get('source') for d in raw_docs[:15]]

        if not isinstance(relevant_files, list): relevant_files = []

        
        # 4. Reconstrucción Contexto (ENRIQUECIDO)
        final_docs = [d for d in raw_docs if d.metadata.get('source') in relevant_files]
        if not final_docs: final_docs = raw_docs[:5] 

        context_lines = []
        for d in final_docs:
            src = d.metadata.get('source')
            otype = d.metadata.get('ontology_type', 'UNKNOWN') # Nuevo campo
            content = d.page_content[:450].replace('\n', ' ')
            # Inyectamos el TIPO en el texto que lee el LLM
            context_lines.append(f"- FILE: {src} [TYPE: {otype}] | CONTENT: {content}...")
            
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