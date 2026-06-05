#!/usr/bin/env python3
"""
run_indexacion.py - Script lanzador para indexación de ontologías.

Responsabilidades:
  - Configurar rutas físicas reales de entrada y salida
  - Instanciar OntologyIndexer con dependencias inyectadas
  - Ejecutar el pipeline de indexación
  - Manejo de errores y logging

Uso:
    python run_indexacion.py
"""

import os
import sys
import logging
from pathlib import Path

# Ajustamos el path para importar la librería
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from ontology_rag import OntologyIndexer


def setup_logging() -> logging.Logger:
    """Configura logging detallado para el script."""
    logger = logging.getLogger("OntologyIndexationScript")
    
    # Handler para consola
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    return logger


def main():
    """Punto de entrada principal."""
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("CONFIGURACIÓN DE INDEXACIÓN DE ONTOLOGÍAS")
    logger.info("=" * 70)
    
    # === CONFIGURACIÓN: Rutas físicas reales ===
    
    # Directorios de origen (ontologías a indexar)
    source_directories = [
        os.path.join(project_root, "dataset"),
        os.path.join(project_root, "gov_acad_dataset"),
        os.path.join(project_root, "dataset_noise_industry"),
    ]
    
    # Directorio de persistencia (ChromaDB)
    persist_directory = os.path.join(project_root, "chroma_db")
    
    # Modelo de embeddings
    embedding_model = "BAAI/bge-m3"
    
    # Parámetros de ingesta
    batch_size = 5000
    recreate_db = True  # True = regenera DB completa, False = añade documentos
    
    logger.info(f"Directorios de origen:")
    for src_dir in source_directories:
        exists = "✓" if os.path.exists(src_dir) else "✗"
        logger.info(f"  [{exists}] {src_dir}")
    
    logger.info(f"Directorio de persistencia: {persist_directory}")
    logger.info(f"Modelo de embeddings: {embedding_model}")
    logger.info(f"Tamaño de lotes: {batch_size}")
    logger.info(f"Recrear BD: {recreate_db}")
    
    # === INSTANCIACIÓN CON INYECCIÓN DE DEPENDENCIAS ===
    
    try:
        indexer = OntologyIndexer(
            source_directories=source_directories,
            persist_directory=persist_directory,
            embedding_model=embedding_model,
            batch_size=batch_size,
            logger=logger,
            recreate_db=recreate_db,
        )
        
        logger.info("✓ OntologyIndexer instanciado correctamente")
        
    except Exception as e:
        logger.error(f"✗ Error al instanciar OntologyIndexer: {e}", exc_info=True)
        sys.exit(1)
    
    # === EJECUCIÓN DEL PIPELINE ===
    
    try:
        indexer.build_index()
        logger.info("✓ Indexación completada exitosamente")
        
    except Exception as e:
        logger.error(f"✗ Error durante indexación: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
