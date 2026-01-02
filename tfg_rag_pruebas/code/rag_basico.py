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
    # En rag_basico.py, dentro de la clase OntologyRecommender

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
        
        # --- NUEVO: WARMUP AUTOMÁTICO ---
        self._warmup_system()
        # --------------------------------
        
        print("Sistema RAG Híbrido listo y caliente (Warmup completado).")

    def _warmup_system(self):
        """Ejecuta una inferencia dummy para cargar modelos en VRAM"""
        print("   🔥 Ejecutando Warmup (Cargando modelos en GPU)...")
        try:
            # 1. Calentar Embeddings
            self.embeddings.embed_query("warmup query")
            
            # 2. Calentar LLM (Forzamos una generación corta)
            # Usamos una invoke directa simple para despertar a Ollama
            self.llm.invoke("Hello, are you ready?")
            print("   🔥 Modelos cargados.")
        except Exception as e:
            print(f"   ⚠️ Error en Warmup (no crítico): {e}")

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
            search_kwargs={"k": 25}
        )


    def _hybrid_retrieve(self, query, k=40):
        """
        Recuperación híbrida: Dense (Chroma) + Sparse (BM25)
        Combina resultados priorizando semántica pero manteniendo keywords exactas.
        """
        # 1. Configurar k dinámicamente para esta llamada
        # Asignamos el mismo k a ambos para asegurar amplitud
        self.bm25_retriever.k = k 
        self.chroma_retriever.search_kwargs["k"] = k # Truco para actualizar Chroma en caliente

        # 2. Recuperación
        dense_docs = self.chroma_retriever.invoke(query)
        sparse_docs = self.bm25_retriever.invoke(query)

        # 3. Combinación y Deduplicación (RRF simplificado o Union)
        # Como ya hemos pedido 'k' a cada uno, no necesitamos recortar aquí inmediatamente
        # si queremos asegurar diversidad.
        
        seen = set()
        combined = []

        # Intercalamos resultados: 1 dense, 1 sparse, 1 dense...
        # Esto ayuda a que no dominen solo los semánticos si k es muy grande.
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
            
            # Parar si ya tenemos k resultados totales
            if len(combined) >= k:
                break

        return combined


    def _setup_chains(self):
        # 1. EXTRACCIÓN Y EXPANSIÓN (Query Expansion - AGNÓSTICO)
        # Objetivo: Enriquecer vocabulario sin asumir el dominio.
        extract_tmpl = """
        Actúa como un terminólogo experto en Web Semántica (OWL/RDF).
        Analiza la petición del usuario: "{user_request}"
        
        Genera una lista de búsqueda optimizada siguiendo estos pasos:
        1. **Conceptos Nucleares:** Extrae los sustantivos y verbos técnicos principales.
        2. **Normalización Ontológica:** Añade los equivalentes formales más probables en ontologías estándar (ej: si dice "tipo", añade "Category", "Class", "Type").
        3. **Sinónimos Técnicos:** Incluye términos alternativos precisos, pero SOLO si son comunes en modelado de conocimiento.
        
        Respuesta: Solo la lista de términos separada por comas (en inglés).
        """
        self.extraction_chain = ChatPromptTemplate.from_template(extract_tmpl) | self.llm | StrOutputParser()

        # 2. FILTRADO (Se mantiene igual de robusto)
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

        # 3. DECISIÓN FINAL (RAZONAMIENTO ESTRUCTURADO - SIN SESGOS)
        # ESTA ES LA VERSIÓN MEJORADA CON CRITERIO DE INTENCIONALIDAD FUNCIONAL
        selection_tmpl = """
        Actúa como un Arquitecto de Ontologías Senior. Tu decisión debe basarse puramente en la lógica de diseño de sistemas y la evidencia del texto.
        
        PETICIÓN USUARIO: "{user_request}"
        CANDIDATOS RECUPERADOS (con metadatos de tipo):
        {filtered_context}
        
        ALGORITMO DE DECISIÓN (Ejecuta esto paso a paso):
        
        PASO 1: Análisis de Especificidad de la Query.
        - ¿La query solicita conceptos fundamentales/genéricos (ej: "qué es un proceso", "definir espacio")? -> **Nivel: GENÉRICO**.
        - ¿La query solicita conceptos aplicados a un nicho (ej: "sensores de riego agrícola", "vigas de acero reforzado")? -> **Nivel: ESPECÍFICO**.
        
        PASO 2: Análisis de Carga Lógica e Intencionalidad (CRÍTICO).
        - Analiza los verbos y sustantivos de la query para determinar la complejidad funcional:
          A. **Intención Normativa/Lógica (Compleja):** ¿La query implica reglas, restricciones, obligaciones, permisos, lógica condicional o comportamiento dinámico? (Palabras clave agnósticas: "Constraint", "Rule", "Must", "Duty", "Function", "Process").
          B. **Intención Descriptiva/Estática (Simple):** ¿La query solo busca etiquetar, anotar o describir atributos estáticos de un recurso? (Palabras clave agnósticas: "Title", "Label", "Tag", "Creator", "Metadata").
        
        PASO 3: Evaluación de Cobertura y Definición.
        - Revisa el contenido de texto de cada candidato.
        - Si (A) Intención Normativa: Descarta vocabularios ligeros de anotación (aunque contengan la palabra clave) y busca ontologías que definan Clases para modelar la regla/acción.
        - Si (B) Intención Descriptiva: Aplica el Principio de Parsimonia. Prefiere vocabularios ligeros y estándar sobre modelos complejos que matarían moscas a cañonazos.
        
        PASO 4: Selección Final.
        - Elige el archivo que mejor se alinee con el Nivel (Genérico/Específico) y la Intención (Normativa/Descriptiva).
        
        SALIDA (JSON estricto):
        {{
            "RAZONAMIENTO": "Explica brevemente la distinción entre Intención Normativa vs Descriptiva detectada y por qué el archivo elegido es el adecuado.",
            "ONTOLOGÍA_RECOMENDADA": "nombre_archivo.ext"
        }}
        """
        self.selection_chain = ChatPromptTemplate.from_template(selection_tmpl) | self.llm | StrOutputParser()

    def _extract_json_from_text(self, text):
        """Extrae y parsea el primer bloque JSON encontrado en un texto."""
        try:
            # Buscar el primer bloque { ... } (incluso multilinea)
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            # Si no encuentra patrón, intenta parsear todo el texto
            return json.loads(text)
        except:
            return None

    def run_pipeline(self, user_request, initial_k=25): # Default a 25
        start_time = time.time()
        print(f"--- Inicio Pipeline: {user_request[:50]}... ---")
        
        # 1. Extracción
        try: keywords = self.extraction_chain.invoke({"user_request": user_request})
        except: keywords = user_request
        print(f"Keywords: {keywords}")

        # 2. Retrieval HÍBRIDO
        raw_docs = self._hybrid_retrieve(keywords, k=initial_k)
        print(f"Retrieval: {len(raw_docs)} docs recuperados.")
        
        # 3. Preparar contexto para FILTRADO (OPTIMIZACIÓN CRÍTICA)
        doc_summaries = []
        for d in raw_docs:
            src = d.metadata.get('source', 'unknown')
            otype = d.metadata.get('ontology_type', '?')
            # REDUCCIÓN: 450 chars -> 180 chars. 
            # Suficiente para ver la definición, ahorra 60% de cómputo.
            content = d.page_content[:180].replace('\n', ' ') 
            doc_summaries.append(f"- ID: {src} [{otype}] | TEXT: {content}...")
        
        doc_list_str = "\n".join(doc_summaries)
        
        # 3. Re-ranking (Filtrado con LLM)
        print("Ejecutando Filtro LLM (esto es lo que tardaba)...")
        relevant_files = []
        try:
            # Ahora el prompt es mucho más ligero (~1.5k tokens vs 5k tokens)
            raw_filter_output = self.filter_chain.invoke({
                "user_request": user_request,
                "context_list": doc_list_str
            })
            parsed_json = self._extract_json_from_text(raw_filter_output)
            
            if parsed_json and "relevant_sources" in parsed_json:
                relevant_files = parsed_json["relevant_sources"]
            else:
                relevant_files = [d.metadata.get('source') for d in raw_docs[:10]]
        except Exception as e:
            print(f"Fallback filtro: {e}")
            relevant_files = [d.metadata.get('source') for d in raw_docs[:10]]

        if not isinstance(relevant_files, list): relevant_files = []
        print(f"Docs tras filtro: {len(relevant_files)}")

        # 4. Reconstrucción Contexto (Para la decisión final SÍ damos más texto)
        final_docs = [d for d in raw_docs if d.metadata.get('source') in relevant_files]
        if not final_docs: final_docs = raw_docs[:5] 

        context_lines = []
        for d in final_docs:
            src = d.metadata.get('source')
            otype = d.metadata.get('ontology_type', 'UNKNOWN')
            # Aquí mantenemos 450 o incluso más, porque ya son pocos documentos (top 3-5)
            content = d.page_content[:500].replace('\n', ' ')
            context_lines.append(f"- FILE: {src} [TYPE: {otype}] | CONTENT: {content}...")
            
        context_str = "\n".join(context_lines)

        # 5. Generación
        print("Generando decisión final...")
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
        res = rag.run_pipeline(q)
        print(f"Filtrados: {res['unique_retrieved_sources']}")
        print(f"Respuesta:\n{res['llm_response']}")