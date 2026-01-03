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
from sentence_transformers import CrossEncoder  # <--- NUEVA IMPORTACIÓN

# CONFIGURACIÓN
# Construir ruta absoluta dinámica: carpeta 'chroma_db' hermana de la carpeta 'code'
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../tfg_rag_pruebas/code
project_root = os.path.dirname(current_dir)              # .../tfg_rag_pruebas
PERSIST_DIRECTORY = os.path.join(project_root, "chroma_db")
EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL = "llama3"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2" # <--- MODELO RERANKER

class OntologyRecommender:
    def __init__(self):
        print("Iniciando sistema RAG con Búsqueda Híbrida + Cross-Encoder Reranking...")
        
        # 1. Embeddings y Vector Store
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        if not os.path.exists(PERSIST_DIRECTORY):
            raise FileNotFoundError(f"No se encuentra la BD en {PERSIST_DIRECTORY}")
            
        self.vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY, 
            embedding_function=self.embeddings
        )
        
        # 2. Cargar Cross-Encoder (Re-ranker)
        # Este modelo es mucho más preciso que un LLM para ordenar relevancia
        print(f" - Cargando Reranker ({RERANKER_MODEL})...")
        self.reranker = CrossEncoder(RERANKER_MODEL)

        # 3. Inicializar Retrievers
        self._setup_retrievers()
        
        # 4. Inicializar LLM y Cadenas
        self.llm = ChatOllama(model=LLM_MODEL, temperature=0.0)
        self._setup_chains()
        
        # --- WARMUP AUTOMÁTICO ---
        self._warmup_system()
        print("Sistema RAG Híbrido listo y optimizado.")

    def _warmup_system(self):
        """Ejecuta una inferencia dummy para cargar modelos en VRAM"""
        print("   🔥 Ejecutando Warmup (Cargando modelos en GPU)...")
        try:
            # 1. Calentar Embeddings
            self.embeddings.embed_query("warmup query")
            
            # 2. Calentar Reranker
            self.reranker.predict([["test query", "test document content"]])

            # 3. Calentar LLM
            self.llm.invoke("Ready?")
            print("   🔥 Modelos cargados.")
        except Exception as e:
            print(f"   ⚠️ Error en Warmup (no crítico): {e}")

    def _setup_retrievers(self):
        """Configura el sistema de recuperación híbrida"""
        print(" - Construyendo índice BM25 (esto puede tardar unos segundos)...")
        
        try:
            collection_data = self.vectorstore.get() 
            texts = collection_data['documents']
            metadatas = collection_data['metadatas']
            
            docs = [
                Document(page_content=t, metadata=m) 
                for t, m in zip(texts, metadatas)
            ]
        except Exception as e:
            print(f"Error cargando docs para BM25: {e}")
            docs = []

        if not docs:
            raise ValueError("La base de datos Chroma parece vacía o no se pudo leer para BM25.")

        self.bm25_retriever = BM25Retriever.from_documents(docs)
        self.bm25_retriever.k = 40

        self.chroma_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 25}
        )

    def _hybrid_retrieve(self, query, k=40):
        """
        Recuperación híbrida: Dense (Chroma) + Sparse (BM25)
        """
        self.bm25_retriever.k = k 
        self.chroma_retriever.search_kwargs["k"] = k 

        dense_docs = self.chroma_retriever.invoke(query)
        sparse_docs = self.bm25_retriever.invoke(query)
        
        seen = set()
        combined = []

        max_len = max(len(dense_docs), len(sparse_docs))
        for i in range(max_len):
            if i < len(dense_docs):
                d = dense_docs[i]
                uid = (d.page_content, str(d.metadata))
                if uid not in seen:
                    seen.add(uid)
                    combined.append(d)
            
            if i < len(sparse_docs):
                d = sparse_docs[i]
                uid = (d.page_content, str(d.metadata))
                if uid not in seen:
                    seen.add(uid)
                    combined.append(d)
            
            if len(combined) >= k:
                break

        return combined

    def _setup_chains(self):
        # 1. EXTRACCIÓN (Igual que antes)
        extract_tmpl = """
        Actúa como un terminólogo experto en Web Semántica (OWL/RDF).
        Analiza la petición del usuario: "{user_request}"
        
        Genera una lista de búsqueda optimizada siguiendo estos pasos:
        1. **Conceptos Nucleares:** Extrae los sustantivos y verbos técnicos principales.
        2. **Normalización Ontológica:** Añade los equivalentes formales más probables en ontologías estándar (ej: si dice "tipo", añade "Category", "Class", "Type").
        3. **Sinónimos Técnicos:** Incluye términos alternativos precisos.
        
        Respuesta: Solo la lista de términos separada por comas (en inglés).
        """
        self.extraction_chain = ChatPromptTemplate.from_template(extract_tmpl) | self.llm | StrOutputParser()

        # NOTA: HE ELIMINADO filter_chain PORQUE AHORA USAMOS CROSS-ENCODER

        # 2. DECISIÓN FINAL (Igual que antes pero procesará mejor contexto)
        selection_tmpl = """
        Actúa como un Arquitecto de Ontologías Senior. Tu decisión debe basarse puramente en la lógica de diseño de sistemas y la evidencia del texto.
        
        PETICIÓN USUARIO: "{user_request}"
        CANDIDATOS RECUPERADOS (Top relevantes tras análisis profundo):
        {filtered_context}
        
        ALGORITMO DE DECISIÓN:
        
        PASO 1: Análisis de Especificidad.
        - ¿Query genérica ("qué es un proceso") o específica de nicho ("sensores de riego")?
        
        PASO 2: Análisis de Intencionalidad (CRÍTICO).
        - **Intención Normativa:** ¿Implica reglas, restricciones, "Must", lógica? -> Busca ontologías pesadas/definitorias.
        - **Intención Descriptiva:** ¿Solo busca etiquetar o metadatos? -> Prefiere vocabularios ligeros (Principio de Parsimonia).
        
        PASO 3: Selección Final.
        - Elige el archivo que mejor se alinee con el Nivel y la Intención.
        
        SALIDA (JSON estricto):
        {{
            "RAZONAMIENTO": "Explica brevemente la distinción entre Intención Normativa vs Descriptiva y la elección.",
            "ONTOLOGÍA_RECOMENDADA": "nombre_archivo.ext"
        }}
        """
        self.selection_chain = ChatPromptTemplate.from_template(selection_tmpl) | self.llm | StrOutputParser()

    def run_pipeline(self, user_request, initial_k=100):
        """
        Pipeline optimizado:
        1. Query Expansion
        2. Broad Retrieval (k=100) -> Para maximizar Recall
        3. Cross-Encoder Reranking -> Para maximizar Precision
        4. LLM Selection (Top 10) -> Para razonamiento final
        """
        start_time = time.time()
        print(f"--- Inicio Pipeline: {user_request[:50]}... ---")
        
        # 1. Extracción
        try: keywords = self.extraction_chain.invoke({"user_request": user_request})
        except: keywords = user_request
        print(f"Keywords: {keywords}")

        # 2. Retrieval HÍBRIDO AMPLIO (k=100)
        # Traemos muchos documentos para evitar que se nos escape el bueno
        raw_docs = self._hybrid_retrieve(keywords, k=initial_k)
        print(f"Retrieval Broad: {len(raw_docs)} docs candidatos.")
        
        # 3. RE-RANKING CON CROSS-ENCODER (El paso crítico)
        print("Ejecutando Cross-Encoder Re-ranking...")
        if raw_docs:
            # Preparamos pares [Query, Doc Content]
            # Limitamos contenido a 500 chars para velocidad del reranker
            doc_contents = [d.page_content[:500] for d in raw_docs]
            pairs = [[user_request, content] for content in doc_contents]
            
            # Predecimos scores de similitud
            scores = self.reranker.predict(pairs)
            
            # Combinamos doc con score y ordenamos
            scored_docs = list(zip(raw_docs, scores))
            scored_docs_sorted = sorted(scored_docs, key=lambda x: x[1], reverse=True)
            
            # Cortamos el Top 10 (High Precision)
            top_k_reranked = 10
            final_docs = [doc for doc, score in scored_docs_sorted[:top_k_reranked]]
            
            top_score = scored_docs_sorted[0][1]
            print(f"Top {top_k_reranked} seleccionados (Score máx: {top_score:.4f})")
        else:
            final_docs = []
            print("⚠️ No se recuperaron documentos en la fase inicial.")

        # 4. Preparar Contexto para el LLM
        context_lines = []
        for d in final_docs:
            src = d.metadata.get('source', 'unknown')
            otype = d.metadata.get('ontology_type', '?')
            # Podemos dar más contexto (600 chars) porque son pocos documentos
            content = d.page_content[:600].replace('\n', ' ')
            context_lines.append(f"- FILE: {src} [TYPE: {otype}] | CONTENT: {content}...")
            
        context_str = "\n".join(context_lines)

        # 5. Generación Final (CoT)
        print("Generando decisión final con LLM...")
        decision_text = self.selection_chain.invoke({
            "user_request": user_request,
            "filtered_context": context_str
        })

        total_time = time.time() - start_time
        print(f"--- Fin Pipeline ({total_time:.2f}s) ---")

        return {
            "query": user_request,
            "keywords": keywords,
            # Devolvemos los filtrados por el Reranker para evaluar el Recall real
            "unique_retrieved_sources": list(set([d.metadata.get('source') for d in final_docs])),
            "llm_response": decision_text,
            "execution_time": total_time
        }

if __name__ == "__main__":
    rag = OntologyRecommender()
    while True:
        q = input("\nConsulta ('salir'): ")
        if q == 'salir': break
        try:
            res = rag.run_pipeline(q)
            print(f"Filtrados (Top-10 Reranked): {res['unique_retrieved_sources']}")
            print(f"Respuesta:\n{res['llm_response']}")
        except Exception as e:
            print(f"Error: {e}")