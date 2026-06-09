import argparse
import sys
import logging
import os
from typing import List

from .creador_rag import OntologyIndexer
from .rag_basico import OntologyRecommender
from .evaluador_requisitos import EvaluadorRequisitos

def setup_logging(debug: bool):
    level = logging.DEBUG if debug else logging.INFO
    
    # 1. Configurar handlers y formato para nuestro logger específico
    formatter = logging.Formatter(
        fmt="[%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    # Logger principal de la librería
    lib_logger = logging.getLogger("ontology_rag")
    lib_logger.setLevel(level)
    lib_logger.addHandler(handler)
    lib_logger.propagate = False
    
    # Logger del CLI
    cli_logger = logging.getLogger("ontology_rag.cli")
    cli_logger.setLevel(level)
    cli_logger.addHandler(handler)
    cli_logger.propagate = False

    # 2. Silenciar el logger raíz y librerías externas ruidosas
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR) # Muta avisos de token/descarga
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)



def handle_index(args):
    logger = logging.getLogger("ontology_rag.cli")
    logger.info("Iniciando indexación de ontologías...")
    
    if not args.src:
        logger.error("Debe proporcionar al menos un directorio de origen con --src")
        sys.exit(1)
        
    try:
        indexer = OntologyIndexer(
            source_directories=args.src,
            persist_directory=args.db,
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            recreate_db=not args.no_recreate,
        )
        indexer.build_index()
        logger.info(f"Indexación completada exitosamente. Base de datos guardada en: {args.db}")
    except Exception as e:
        logger.error(f"Error durante la indexación: {e}", exc_info=args.debug)
        sys.exit(1)

def handle_query(args):
    logger = logging.getLogger("ontology_rag.cli")
    logger.info(f"Ejecutando consulta RAG para: '{args.question}'")
    
    try:
        recommender = OntologyRecommender(
            persist_directory=args.db,
            embedding_model=args.embedding_model,
            llm_model=args.llm_model,
            reranker_model=args.reranker_model,
            temperature=args.temp,
            warmup=args.warmup,
        )
        response = recommender.run_pipeline(args.question)
        print("\n=== RESPUESTA DEL RECOMENDADOR ===")
        print(response.get("llm_response", ""))
        print("===================================\n")
        logger.info(f"Ontologías recuperadas: {response.get('unique_retrieved_sources', [])}")
        logger.info(f"Tiempo de ejecución: {response.get('execution_time', 0.0):.2f}s")
    except Exception as e:
        logger.error(f"Error ejecutando la consulta: {e}", exc_info=args.debug)
        sys.exit(1)

def handle_evaluate(args):
    logger = logging.getLogger("ontology_rag.cli")
    logger.info(f"Lanzando evaluación de requisitos desde CSV: {args.csv}")
    
    selected_ids = None
    if args.selected_ids:
        try:
            selected_ids = [int(x.strip()) for x in args.selected_ids.split(",") if x.strip().isdigit()]
        except Exception as e:
            logger.error(f"Error procesando IDs seleccionados: {e}")
            sys.exit(1)
            
    try:
        evaluador = EvaluadorRequisitos(
            persist_directory=args.db,
            embedding_model=args.embedding_model,
            llm_model=args.llm_model,
            reranker_model=args.reranker_model,
            warmup=args.warmup,
            output_dir=args.output,
        )
        evaluador.orquestar_evaluacion(
            ruta_csv=args.csv,
            max_requirements=args.max_reqs,
            selected_ids=selected_ids,
        )
        logger.info("Evaluación completada con éxito.")
    except Exception as e:
        logger.error(f"Error ejecutando la evaluación: {e}", exc_info=args.debug)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="CLI para la librería modular de recuperación y evaluación de ontologías (Ontology RAG Evaluator)"
    )
    parser.add_argument("--debug", action="store_true", help="Activar trazas detalladas de depuración (DEBUG)")
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcomandos disponibles")
    
    # Subcomando: index
    p_index = subparsers.add_parser("index", help="Indexar y vectorizar ontologías hacia ChromaDB")
    p_index.add_argument("--src", nargs="+", required=True, help="Directorios de origen con archivos de ontologías RDF/TTL/OWL")
    p_index.add_argument("--db", default="chroma_db", help="Directorio de destino para persistir ChromaDB (por defecto: 'chroma_db')")
    p_index.add_argument("--embedding-model", default="BAAI/bge-m3", help="Modelo de embeddings a utilizar (por defecto: 'BAAI/bge-m3')")
    p_index.add_argument("--batch-size", type=int, default=5000, help="Tamaño de lotes para el procesamiento (por defecto: 5000)")
    p_index.add_argument("--no-recreate", action="store_true", help="Evitar borrar la base de datos existente (añadir en lugar de recrear)")
    
    # Subcomando: query
    p_query = subparsers.add_parser("query", help="Realizar una consulta RAG individual")
    p_query.add_argument("question", help="Requisito funcional o pregunta en lenguaje natural")
    p_query.add_argument("--db", default="chroma_db", help="Directorio donde está guardada ChromaDB (por defecto: 'chroma_db')")
    p_query.add_argument("--embedding-model", default="BAAI/bge-m3", help="Modelo de embeddings (por defecto: 'BAAI/bge-m3')")
    p_query.add_argument("--llm-model", default="llama3", help="Modelo LLM de Ollama (por defecto: 'llama3')")
    p_query.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="Modelo de re-ranking (por defecto: 'cross-encoder/ms-marco-MiniLM-L-6-v2')")
    p_query.add_argument("--temp", type=float, default=0.0, help="Temperatura del LLM (por defecto: 0.0)")
    p_query.add_argument("--no-warmup", dest="warmup", action="store_false", help="Desactivar warmup (precalentamiento de modelos en GPU)")
    p_query.set_defaults(warmup=True)
    
    # Subcomando: evaluate
    p_eval = subparsers.add_parser("evaluate", help="Ejecutar el benchmark completo sobre un CSV de requisitos")
    p_eval.add_argument("--csv", required=True, help="Ruta al archivo CSV con los requisitos")
    p_eval.add_argument("--db", default="chroma_db", help="Directorio donde está guardada ChromaDB (por defecto: 'chroma_db')")
    p_eval.add_argument("--embedding-model", default="BAAI/bge-m3", help="Modelo de embeddings (por defecto: 'BAAI/bge-m3')")
    p_eval.add_argument("--llm-model", default="llama3", help="Modelo LLM de Ollama (por defecto: 'llama3')")
    p_eval.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="Modelo de re-ranking (por defecto: 'cross-encoder/ms-marco-MiniLM-L-6-v2')")
    p_eval.add_argument("--no-warmup", dest="warmup", action="store_false", help="Desactivar warmup (precalentamiento de modelos en GPU)")
    p_eval.set_defaults(warmup=True)
    p_eval.add_argument("--output", help="Directorio para guardar los informes y trazas de salida (por defecto: './resultado')")
    p_eval.add_argument("--max-reqs", type=int, help="Número máximo de requisitos a evaluar (opcional)")
    p_eval.add_argument("--selected-ids", help="Lista de IDs de requisitos separados por comas a evaluar (ej. '1,5,12')")
    
    args = parser.parse_args()
    
    setup_logging(args.debug)
    
    if args.command == "index":
        handle_index(args)
    elif args.command == "query":
        handle_query(args)
    elif args.command == "evaluate":
        handle_evaluate(args)

if __name__ == "__main__":
    main()
