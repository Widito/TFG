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
from sentence_transformers import CrossEncoder  # NUEVA IMPORTACIÓN

# CONFIGURACIÓN
# Construir ruta absoluta dinámica
current_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.dirname(current_dir)              
PERSIST_DIRECTORY = os.path.join(project_root, "chroma_db")
EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL = "llama3"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2" # MODELO RERANKER



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
        print("   Ejecutando Warmup (Cargando modelos en GPU)...")
        try:
            self.embeddings.embed_query("warmup query")
            self.reranker.predict([["test query", "test document content"]])
            self.llm.invoke("Ready?")
            print("   Modelos cargados.")
        except Exception as e:
            print(f"   Error en Warmup (no crítico): {e}")

    def _setup_retrievers(self):
        print(" - Construyendo índice BM25...")
        try:
            collection_data = self.vectorstore.get() 
            texts = collection_data['documents']
            metadatas = collection_data['metadatas']
            docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
        except Exception as e:
            print(f"Error cargando docs para BM25: {e}")
            docs = []

        if not docs: raise ValueError("BD Chroma vacía.")
        self.bm25_retriever = BM25Retriever.from_documents(docs)
        self.bm25_retriever.k = 40
        self.chroma_retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 25})

    def _hybrid_retrieve(self, query, k=40):
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
                if uid not in seen: seen.add(uid); combined.append(d)
            if i < len(sparse_docs):
                d = sparse_docs[i]
                uid = (d.page_content, str(d.metadata))
                if uid not in seen: seen.add(uid); combined.append(d)
            if len(combined) >= k: break
        return combined

    def _setup_chains(self):
        # 1. EXTRACCIÓN
        extract_tmpl = """
        Actúa como un terminólogo experto en Web Semántica (OWL/RDF).
        Analiza la petición del usuario: "{user_request}"
        
        Genera una lista de búsqueda optimizada siguiendo estos pasos:
        1. **Conceptos Nucleares:** Extrae los sustantivos y verbos técnicos principales.
        2. **Normalización Ontológica:** Añade los equivalentes formales más probables (Category, Class, Type).
        3. **Sinónimos Técnicos:** Incluye términos alternativos precisos.
        
        Respuesta: Solo la lista de términos separada por comas (en inglés).
        """
        self.extraction_chain = ChatPromptTemplate.from_template(extract_tmpl) | self.llm | StrOutputParser()

        # 2. DECISIÓN FINAL (PROMPT REFINADO PARA LA NUEVA ESTRATEGIA)
        selection_tmpl = """
        Actúa como un Arquitecto de Ontologías Senior. Tu decisión debe basarse puramente en la lógica de diseño de sistemas y la evidencia del texto.
        
        PETICIÓN USUARIO: "{user_request}"
        
        CANDIDATOS RECUPERADOS (Mejores coincidencias):
        {filtered_context}
        
        ALGORITMO DE DECISIÓN:
        1. **Análisis de Especificidad:** ¿La query es genérica (Conceptos abstractos) o específica (Dominio concreto)?
        2. **Análisis de Intencionalidad:**
           - Si busca **Reglas/Restricciones** ("Must", "Policy"), prioriza ontologías normativas (ej: ODRL).
           - Si busca **Definiciones Estructurales** ("Building", "Sensor"), prioriza ontologías de dominio (ej: BOT, SSN).
        3. **Desambiguación:**
           - Si múltiples archivos parecen válidos (ej: 'building.ttl' vs 'bot.ttl'), PREFIERE el estándar reconocido (W3C/ETSI) o el que defina explícitamente la Clase principal solicitada.
        
        SALIDA (JSON estricto):
        {{
            "RAZONAMIENTO": "Breve explicación de por qué el archivo elegido encaja mejor con la intención.",
            "ONTOLOGÍA_RECOMENDADA": "nombre_archivo.ext"
        }}
        """
        self.selection_chain = ChatPromptTemplate.from_template(selection_tmpl) | self.llm | StrOutputParser()

    # --- NUEVO MÉTODO PARA MEJORAR SNIPPETS ---
    def _extract_rich_snippet(self, text, max_len=1200):
        """
        Intenta encontrar el inicio de las definiciones relevantes (Classes/Properties)
        saltando los metadatos/licencias iniciales si es posible.
        """
        # Marcadores típicos de definición en Turtle/N3
        markers = [" a owl:Class", " a rdfs:Class", " a owl:ObjectProperty", " a owl:DatatypeProperty", " a skos:Concept"]
        
        start_idx = -1
        
        # Buscamos el primer marcador que aparezca
        for marker in markers:
            idx = text.find(marker)
            if idx != -1:
                # Si encontramos un marcador, intentamos retroceder un poco para coger el sujeto
                # (ej: "bot:Zone a owl:Class" -> queremos pillar "bot:Zone")
                candidate_start = max(0, idx - 50) 
                
                # Nos quedamos con el marcador que aparezca antes en el texto
                if start_idx == -1 or candidate_start < start_idx:
                    start_idx = candidate_start
        
        # Si no encontramos marcadores claros, o están muy al principio, usamos el inicio normal
        # (A veces el inicio es importante si tiene comments descriptivos generales)
        if start_idx == -1 or start_idx < 200:
            clean_text = text[:max_len]
        else:
            # Si el marcador está muy lejos (ej: línea 500), empezamos ahí para saltar la licencia
            print(f"   (Snippet optimizado: saltando {start_idx} chars de encabezado)")
            clean_text = "..." + text[start_idx : start_idx + max_len]
            
        return clean_text.replace('\n', ' ')

    def run_pipeline(self, user_request, initial_k=100):
        start_time = time.time()
        print(f"--- Inicio Pipeline: {user_request[:50]}... ---")
        
        # 1. Extracción
        try: keywords = self.extraction_chain.invoke({"user_request": user_request})
        except: keywords = user_request

        # 2. Retrieval HÍBRIDO AMPLIO
        raw_docs = self._hybrid_retrieve(keywords, k=initial_k)
        print(f"Retrieval Broad: {len(raw_docs)} docs candidatos.")
        
        # 3. RE-RANKING
        print("Ejecutando Cross-Encoder Re-ranking...")
        if raw_docs:
            doc_contents = [d.page_content[:500] for d in raw_docs] # Para reranker usamos inicio (suele bastar)
            pairs = [[user_request, content] for content in doc_contents]
            scores = self.reranker.predict(pairs)
            
            scored_docs = list(zip(raw_docs, scores))
            scored_docs_sorted = sorted(scored_docs, key=lambda x: x[1], reverse=True)
            
            # Top 10 High Quality
            top_k_reranked = 10
            final_docs = [doc for doc, score in scored_docs_sorted[:top_k_reranked]]
            print(f"Top {top_k_reranked} seleccionados (Score máx: {scored_docs_sorted[0][1]:.4f})")
        else:
            final_docs = []

        # 4. Preparar Contexto RICO para el LLM
        context_lines = []
        for d in final_docs:
            src = d.metadata.get('source', 'unknown')
            otype = d.metadata.get('ontology_type', '?')
            
            # --- USAMOS EL NUEVO EXTRACTOR ---
            # Extraemos 1200 chars centrados en definiciones, no en licencias
            content = self._extract_rich_snippet(d.page_content, max_len=1200)
            # ---------------------------------
            
            context_lines.append(f"- FILE: {src} [TYPE: {otype}] | CONTENT: {content}...")
            
        context_str = "\n".join(context_lines)

        # 5. Generación Final
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