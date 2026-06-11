import json
import os
import re
import time
import logging
from typing import Optional

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class OntologyRecommender:
    def __init__(
        self,
        persist_directory: str = "chroma_db",
        embedding_model: str = "BAAI/bge-m3",
        llm_model: str = "llama3",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        temperature: float = 0.0,
        warmup: bool = True,
        llm: Optional[BaseChatModel] = None,
    ):
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.reranker_model = reranker_model
        self.temperature = temperature

        logger.info("Iniciando sistema RAG con Búsqueda Híbrida + Cross-Encoder Reranking...")

        # Intentar cargar embeddings en local, si falla descargar en primer arranque
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"local_files_only": True}
            )
        except Exception:
            logger.info("Modelo de embeddings no encontrado localmente. Descargando desde Hugging Face...")
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)

        if not os.path.exists(self.persist_directory):
            raise FileNotFoundError(f"No se encuentra la BD en {self.persist_directory}")

        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )

        logger.info(f" - Cargando Reranker ({self.reranker_model})...")
        
        # Intentar cargar reranker en local, si falla descargar en primer arranque
        try:
            self.reranker = CrossEncoder(self.reranker_model, local_files_only=True)
        except Exception:
            logger.info("Modelo Reranker no encontrado localmente. Descargando desde Hugging Face...")
            self.reranker = CrossEncoder(self.reranker_model)


        self._setup_retrievers()

        if llm is not None:
            self.llm = llm
        else:
            self.llm = ChatOllama(model=self.llm_model, temperature=self.temperature)
        self._setup_chains()

        if warmup:
            self._warmup_system()

        logger.info("Sistema RAG Híbrido listo y optimizado.")

    def _warmup_system(self):
        """Ejecuta una inferencia dummy para cargar modelos en VRAM"""
        logger.info("   Ejecutando Warmup (Cargando modelos en GPU)...")
        try:
            self.embeddings.embed_query("warmup query")
            self.reranker.predict([["test query", "test document content"]])
            self.llm.invoke("Ready?")
            logger.info("   Modelos cargados.")
        except Exception as e:
            logger.warning(f"   Error en Warmup (no crítico): {e}")

    def _setup_retrievers(self):
        logger.info(" - Construyendo índice BM25...")
        try:
            collection_data = self.vectorstore.get()
            texts = collection_data.get("documents") or []
            metadatas = collection_data.get("metadatas") or []
            docs = [Document(page_content=t, metadata=m or {}) for t, m in zip(texts, metadatas)]
        except Exception as e:
            logger.error(f"Error cargando docs para BM25: {e}")
            docs = []

        if not docs:
            raise ValueError("BD Chroma vacía.")

        self.bm25_retriever = BM25Retriever.from_documents(docs)
        self.bm25_retriever.k = 40
        self.chroma_retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 25})

    def _hybrid_retrieve(self, query, k=40, retrieval_mode="hybrid"):
        self.bm25_retriever.k = k

        uris_vistas = set()
        unique_docs = []
        unique_metadatas = []
        unique_distances = []

        if retrieval_mode in ("dense", "hybrid"):
            query_embedding = self.embeddings.embed_query(query)
            raw_results = self.vectorstore._collection.query(
                query_embeddings=[query_embedding],
                n_results=k * 3,
                include=["documents", "metadatas", "distances"],
            )

            documents_batch = raw_results.get("documents") or [[]]
            metadatas_batch = raw_results.get("metadatas") or [[]]
            distances_batch = raw_results.get("distances") or [[]]

            dense_documents = documents_batch[0] if documents_batch else []
            dense_metadatas = metadatas_batch[0] if metadatas_batch else []
            dense_distances = distances_batch[0] if distances_batch else []

            for document_text, metadata, distance in zip(dense_documents, dense_metadatas, dense_distances):
                metadata = metadata or {}
                uri = str(metadata.get("uri", "")).strip()
                if not uri:
                    match = re.search(r"^URI:\s*(\S+)", str(document_text or ""), flags=re.MULTILINE)
                    uri = match.group(1).strip() if match else ""

                if uri in uris_vistas:
                    continue

                uris_vistas.add(uri)
                unique_docs.append(Document(page_content=document_text, metadata=metadata))
                unique_metadatas.append(metadata)
                unique_distances.append(distance)

                if len(unique_docs) >= k:
                    break

        if retrieval_mode in ("bm25", "hybrid") and len(unique_docs) < k:
            sparse_docs = self.bm25_retriever.invoke(query)
            for doc in sparse_docs:
                uri = str((doc.metadata or {}).get("uri", "")).strip()
                if not uri:
                    match = re.search(r"^URI:\s*(\S+)", str(doc.page_content or ""), flags=re.MULTILINE)
                    uri = match.group(1).strip() if match else ""

                if uri in uris_vistas:
                    continue

                uris_vistas.add(uri)
                unique_docs.append(doc)
                unique_metadatas.append(doc.metadata)
                unique_distances.append(None)

                if len(unique_docs) >= k:
                    break

        return unique_docs

    def _setup_chains(self):
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

    def _extract_rich_snippet(self, text, max_len=1200):
        markers = [" a owl:Class", " a rdfs:Class", " a owl:ObjectProperty", " a owl:DatatypeProperty", " a skos:Concept"]
        start_idx = -1

        for marker in markers:
            idx = text.find(marker)
            if idx != -1:
                candidate_start = max(0, idx - 50)
                if start_idx == -1 or candidate_start < start_idx:
                    start_idx = candidate_start

        if start_idx == -1 or start_idx < 200:
            clean_text = text[:max_len]
        else:
            logger.debug(f"   (Snippet optimizado: saltando {start_idx} chars de encabezado)")
            clean_text = "..." + text[start_idx : start_idx + max_len]

        return clean_text.replace("\n", " ")

    def run_pipeline(self, user_request, initial_k=100, retrieval_mode="hybrid", use_reranker=True):
        start_time = time.time()
        logger.info(f"--- Inicio Pipeline: {user_request[:50]}... ---")

        try:
            keywords = self.extraction_chain.invoke({"user_request": user_request})
        except Exception:
            keywords = user_request

        raw_docs = self._hybrid_retrieve(keywords, k=initial_k, retrieval_mode=retrieval_mode)
        logger.info(f"Retrieval Broad: {len(raw_docs)} docs candidatos.")

        logger.info("Ejecutando Cross-Encoder Re-ranking...")
        if raw_docs:
            if use_reranker:
                doc_contents = [d.page_content[:500] for d in raw_docs]
                pairs = [[user_request, content] for content in doc_contents]
                scores = self.reranker.predict(pairs)

                scored_docs = list(zip(raw_docs, scores))
                scored_docs_sorted = sorted(scored_docs, key=lambda x: x[1], reverse=True)

                top_k_reranked = 10
                final_docs = [doc for doc, score in scored_docs_sorted[:top_k_reranked]]
                logger.info(f"Top {top_k_reranked} seleccionados (Score máx: {scored_docs_sorted[0][1]:.4f})")
            else:
                top_k_no_rerank = 10
                final_docs = raw_docs[:top_k_no_rerank]
                logger.info(f"Top {top_k_no_rerank} seleccionados (Sin re-ranking)")
        else:
            final_docs = []

        context_lines = []
        for d in final_docs:
            src = d.metadata.get("source", "unknown")
            otype = d.metadata.get("ontology_type", "?")
            content = self._extract_rich_snippet(d.page_content, max_len=1200)
            context_lines.append(f"- FILE: {src} [TYPE: {otype}] | CONTENT: {content}...")

        context_str = "\n".join(context_lines)

        logger.info("Generando decisión final con LLM...")
        decision_text = self.selection_chain.invoke({
            "user_request": user_request,
            "filtered_context": context_str,
        })

        total_time = time.time() - start_time
        logger.info(f"--- Fin Pipeline ({total_time:.2f}s) ---")

        return {
            "query": user_request,
            "keywords": keywords,
            "unique_retrieved_sources": list(set([d.metadata.get("source") for d in final_docs])),
            "llm_response": decision_text,
            "execution_time": total_time,
        }

