"""
OntologyIndexer - Módulo de ingesta y vectorización de ontologías.

Responsabilidades:
  - Parsing robusta de archivos RDF/Turtle/N3/OWL
  - Análisis estructural de ontologías (CORE vs EXTENSION)
  - Extracción de Clases y Propiedades mediante SPARQL
  - Fragmentación (chunking) semántica de documentos
  - Vectorización con HuggingFace Embeddings
  - Persistencia en ChromaDB con metadatos estructurales

Patrón: Inyección de Dependencias + OOP
"""

import os
import shutil
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import rdflib
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


class OntologyIndexer:
    """
    Indexador modular de ontologías con inyección de dependencias.
    
    Soporta múltiples fuentes y formatos RDF, generando vectores
    en ChromaDB con metadatos estructurales para búsqueda híbrida.
    """
    
    FORMAT_MAP = {
        '.ttl': 'turtle',
        '.n3': 'n3',
        '.owl': 'xml',
        '.rdf': 'xml',
        '.nt': 'nt'
    }
    
    SPARQL_CLASSES = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX dc: <http://purl.org/dc/elements/1.1/>
        SELECT ?uri ?label ?comment ?def_skos WHERE {
            { ?uri a rdfs:Class . } UNION { ?uri a owl:Class . }
            FILTER(!isblank(?uri))
            OPTIONAL { ?uri rdfs:label ?label . }
            OPTIONAL { ?uri rdfs:comment ?comment . }
            OPTIONAL { ?uri skos:definition ?def_skos . }
        }
    """
    
    SPARQL_PROPERTIES = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        SELECT ?uri ?label ?comment ?def_skos ?domain ?range WHERE {
            { ?uri a rdf:Property . } UNION { ?uri a owl:ObjectProperty . } UNION { ?uri a owl:DatatypeProperty . }
            FILTER(!isblank(?uri))
            OPTIONAL { ?uri rdfs:label ?label . }
            OPTIONAL { ?uri rdfs:comment ?comment . }
            OPTIONAL { ?uri skos:definition ?def_skos . }
            OPTIONAL { ?uri rdfs:domain ?domain . }
            OPTIONAL { ?uri rdfs:range ?range . }
        }
    """
    
    def __init__(
        self,
        source_directories: List[str],
        persist_directory: str,
        embedding_model: str = "BAAI/bge-m3",
        batch_size: int = 5000,
        logger: Optional[logging.Logger] = None,
        recreate_db: bool = True,
    ):
        """
        Inicializa el indexador con inyección de dependencias.
        
        Args:
            source_directories: Lista de rutas con ontologías a indexar
            persist_directory: Ruta donde guardar ChromaDB
            embedding_model: Modelo de HuggingFace para embeddings
            batch_size: Tamaño de lotes para procesamiento
            logger: Logger (opcional, crea uno por defecto si no se proporciona)
            recreate_db: Si True, elimina y recrea la BD; si False, añade documentos
        """
        self.source_directories = source_directories
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.recreate_db = recreate_db
        
        self.logger = logger or self._setup_logger()
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"local_files_only": True}
            )
        except Exception:
            self.logger.info("Modelo de embeddings no encontrado localmente. Descargando desde Hugging Face...")
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.vectorstore = None
    
    def _setup_logger(self) -> logging.Logger:
        """Configura logging por defecto."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    @staticmethod
    def _get_safe_value(row, attr_list: List[str]) -> str:
        """Extrae valor seguro de propiedades RDF."""
        for attr in attr_list:
            if hasattr(row, attr) and getattr(row, attr):
                return str(getattr(row, attr))
        return ""
    
    @staticmethod
    def _analyze_ontology_structure(graph: rdflib.Graph) -> Tuple[str, int]:
        """
        Analiza la estructura del grafo para clasificar la ontología.
        
        Returns:
            Tupla (ontology_type, import_count) donde ontology_type es 'CORE' o 'EXTENSION'
        """
        query_imports = "SELECT (COUNT(?o) AS ?count) WHERE { ?s <http://www.w3.org/2002/07/owl#imports> ?o }"
        try:
            res = list(graph.query(query_imports))
            import_count = int(res[0][0])
        except Exception:
            import_count = 0
        
        # Heurística: Importaciones indican extensión, caso contrario es CORE
        ontology_type = "EXTENSION" if import_count > 0 else "CORE"
        return ontology_type, import_count
    
    def _parse_ontology_file(
        self, 
        filepath: str,
        filename: str,
        origin_folder: str
    ) -> Tuple[List[str], List[Dict]]:
        """
        Parsea un archivo RDF y extrae Clases y Propiedades.
        
        Args:
            filepath: Ruta completa al archivo
            filename: Nombre del archivo
            origin_folder: Carpeta de origen (para metadata)
            
        Returns:
            Tupla (documents, metadatas)
        """
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in self.FORMAT_MAP:
            return [], []
        
        graph = rdflib.Graph()
        loaded = False
        
        # Carga robusta con fallback
        try:
            graph.parse(filepath, format=self.FORMAT_MAP[file_ext])
            loaded = True
        except Exception:
            if self.FORMAT_MAP[file_ext] == 'xml':
                try:
                    graph.parse(filepath, format='turtle')
                    loaded = True
                except Exception:
                    pass
        
        if not loaded:
            self.logger.warning(f"No se pudo cargar: {filepath}")
            return [], []
        
        # Análisis estructural
        ont_type, n_imports = self._analyze_ontology_structure(graph)
        
        documents = []
        metadatas = []
        
        # Procesar Clases
        try:
            for row in graph.query(self.SPARQL_CLASSES):
                label = self._get_safe_value(row, ['label']) or str(row.uri).split('#')[-1]
                desc = self._get_safe_value(row, ['comment', 'def_skos'])
                
                doc_text = (
                    f"Concept: Class\n"
                    f"Ontology: {filename} ({ont_type})\n"
                    f"URI: {row.uri}\n"
                    f"Label: {label}\n"
                    f"Definition: {desc}"
                )
                documents.append(doc_text)
                
                metadatas.append({
                    "source": filename,
                    "ontology_type": ont_type,
                    "imports_count": n_imports,
                    "origin_folder": origin_folder,
                    "uri": str(row.uri),
                    "concept_type": "Class",
                })
        except Exception as e:
            self.logger.error(f"Error procesando clases en {filename}: {e}")
        
        # Procesar Propiedades
        try:
            for row in graph.query(self.SPARQL_PROPERTIES):
                label = self._get_safe_value(row, ['label']) or str(row.uri).split('#')[-1]
                desc = self._get_safe_value(row, ['comment', 'def_skos'])
                
                doc_text = (
                    f"Concept: Property\n"
                    f"Ontology: {filename} ({ont_type})\n"
                    f"URI: {row.uri}\n"
                    f"Label: {label}\n"
                    f"Definition: {desc}"
                )
                documents.append(doc_text)
                
                metadatas.append({
                    "source": filename,
                    "ontology_type": ont_type,
                    "imports_count": n_imports,
                    "origin_folder": origin_folder,
                    "uri": str(row.uri),
                    "concept_type": "Property",
                })
        except Exception as e:
            self.logger.error(f"Error procesando propiedades en {filename}: {e}")
        
        return documents, metadatas
    
    def _extract_from_directories(self) -> Tuple[List[str], List[Dict]]:
        """
        Extrae documentos y metadatos de todos los directorios de origen.
        
        Returns:
            Tupla (all_documents, all_metadatas)
        """
        all_documents = []
        all_metadatas = []
        
        for source_dir in self.source_directories:
            if not os.path.exists(source_dir):
                self.logger.warning(f"Carpeta no encontrada: {source_dir}")
                continue
            
            self.logger.info(f"Procesando directorio: {os.path.basename(source_dir)}")
            
            for filename in os.listdir(source_dir):
                filepath = os.path.join(source_dir, filename)
                if not os.path.isfile(filepath):
                    continue
                
                origin_folder = os.path.basename(source_dir)
                docs, metas = self._parse_ontology_file(filepath, filename, origin_folder)
                
                if docs:
                    all_documents.extend(docs)
                    all_metadatas.extend(metas)
                    self.logger.debug(f"  ✓ {filename}: {len(docs)} fragmentos extraídos")
        
        return all_documents, all_metadatas
    
    def _create_vectorstore(
        self, 
        documents: List[str], 
        metadatas: List[Dict]
    ) -> None:
        """
        Crea o actualiza el vectorstore en ChromaDB.
        
        Args:
            documents: Lista de textos a vectorizar
            metadatas: Lista de diccionarios de metadatos
        """
        if not documents:
            raise ValueError("No hay documentos para indexar.")
        
        self.logger.info(f"Ingesta: {len(documents)} fragmentos hacia ChromaDB...")
        
        # Regenerar DB si se solicita
        if self.recreate_db and os.path.exists(self.persist_directory):
            self.logger.info("Eliminando BD existente para regeneración...")
            shutil.rmtree(self.persist_directory)
        
        # Procesar en lotes para evitar memoria
        for i in range(0, len(documents), self.batch_size):
            batch_start = i
            batch_end = min(i + self.batch_size, len(documents))
            
            self.logger.info(f"  Lote {batch_start}-{batch_end}...")
            
            batch_docs = documents[batch_start:batch_end]
            batch_metas = metadatas[batch_start:batch_end]
            
            if self.vectorstore is None:
                # Primera vez: crear
                self.vectorstore = Chroma.from_texts(
                    texts=batch_docs,
                    embedding=self.embeddings,
                    metadatas=batch_metas,
                    persist_directory=self.persist_directory,
                )
            else:
                # Siguientes: añadir
                self.vectorstore.add_texts(
                    texts=batch_docs,
                    metadatas=batch_metas,
                )
        
        self.logger.info("✓ Ingesta completada con metadatos estructurales.")
    
    def build_index(self) -> None:
        """
        Ejecuta el pipeline completo de indexación:
        1. Extrae documentos de fuentes
        2. Crea/actualiza vectorstore
        """
        self.logger.info("=" * 60)
        self.logger.info("INICIANDO INDEXACIÓN DE ONTOLOGÍAS")
        self.logger.info("=" * 60)
        
        # Extracción
        documents, metadatas = self._extract_from_directories()
        
        if not documents:
            raise ValueError("No se extrajeron documentos. Verifica rutas de origen.")
        
        self.logger.info(f"Total de fragmentos extraídos: {len(documents)}")
        
        # Vectorización
        self._create_vectorstore(documents, metadatas)
        
        self.logger.info("=" * 60)
        self.logger.info("INDEXACIÓN COMPLETADA")
        self.logger.info(f"BD guardada en: {self.persist_directory}")
        self.logger.info("=" * 60)
    
    def get_vectorstore(self) -> Chroma:
        """Retorna el vectorstore para usar en otros componentes."""
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
            )
        return self.vectorstore
