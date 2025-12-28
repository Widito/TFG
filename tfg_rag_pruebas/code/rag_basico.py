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
PERSIST_DIRECTORY = "tfg_rag_pruebas/chroma_db"
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
        # 1. EXTRACCIÓN 
        extract_tmpl = """
        Analiza la petición del usuario.
        Petición: {user_request}
        
        Tarea: Genera una lista de palabras clave técnicas y conceptos principales para buscar.
        - Usa terminología técnica en inglés.
        - Incluye sinónimos si es necesario.
        
        Respuesta: Solo las palabras clave separadas por comas.
        """
        self.extraction_chain = ChatPromptTemplate.from_template(extract_tmpl) | self.llm | StrOutputParser()

        # 2. FILTRADO (Sin cambios mayores, solo robustez)
        filter_tmpl = """
        Eres un filtro de relevancia experto.
        
        INPUT:
        - Query: "{user_request}"
        - Candidatos: Lista de fragmentos de texto.

        TAREA:
        Selecciona TODOS los archivos que contengan definiciones útiles para la query.
        
        CRITERIOS CRÍTICOS:
        1. **Mención Explícita:** Si un archivo contiene la PALABRA EXACTA buscada (ej: "Switch", "Device"), DEBE ser seleccionado, incluso si parece de otro dominio.
        2. **Coherencia de Dominio:** - Si la query es de Construcción -> Busca 'Building', 'Zone', 'Space'.
           - Si la query es de IoT/Dispositivos -> Busca 'Device', 'Sensor', 'Function', 'Command'.
           - SAREF es clave para IoT. BOT es clave para Construcción.

        SALIDA (JSON VÁLIDO):
        {{
            "relevant_sources": ["archivo1.ttl", "archivo2.n3"]
        }}
        """
        self.filter_chain = ChatPromptTemplate.from_template(filter_tmpl) | self.llm | StrOutputParser()

        # 3. DECISIÓN FINAL 
        selection_tmpl = """
        Eres un Sistema Recomendador de Ontologías Experto en IoT y Construcción.
        
        Petición del Usuario: {user_request}
        
        Contexto (Candidatos Filtrados):
        {filtered_context}
        
        INSTRUCCIONES DE SELECCIÓN:
        1. **IoT vs Construcción:**
           - Si el usuario define "Device", "Sensor", "Actuator", "Command", "Function" -> RECOMIENDA ontologías IoT (prioridad: saref, sosa, ssn).
           - Si el usuario define "Zone", "Building", "Storey", "Space" (topología) -> RECOMIENDA ontologías de Construcción (prioridad: bot).
        
        2. **Jerarquía SAREF:**
           - Si la definición parece genérica de un dispositivo (ej: "perform a function"), prefiere el NÚCLEO (saref_xxx.n3) antes que extensiones específicas (s4agri, s4city), a menos que el contexto sea explícitamente agrícola o urbano.

        3. **Formato:**
           - Responde SOLO con el nombre del archivo y la razón.
           - PROHIBIDO URIs.

        Respuesta:
        **ONTOLOGÍA RECOMENDADA:** [NOMBRE_DEL_ARCHIVO]
        **RAZÓN:** [Justificación técnica]
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

        # 4. Reconstrucción Contexto
        final_docs = [d for d in raw_docs if d.metadata.get('source') in relevant_files]
        if not final_docs: final_docs = raw_docs[:5] 

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