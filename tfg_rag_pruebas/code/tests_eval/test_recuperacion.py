import os
import sys
import logging
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


# Configurar sys.path para importar localmente desde src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, os.path.join(project_root, "src"))

from ontology_rag import OntologyRecommender

# Configurar logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("TestRecuperacion")


def sanitize_str(x):
    """Sanitiza una cadena para evitar fallos de codificación Unicode en la terminal de Windows."""
    s = str(x)
    try:
        encoding = sys.stdout.encoding or 'utf-8'
        return s.encode(encoding, errors='replace').decode(encoding)
    except Exception:
        return s.encode('ascii', errors='replace').decode('ascii')


def print_table(headers, data, title=""):
    """Imprime una tabla formateada en consola."""
    # Sanitizar cabeceras y celdas
    headers = [sanitize_str(h) for h in headers]
    data = [[sanitize_str(cell) for cell in row] for row in data]
    
    print(f"\n--- {title} ---")
    # Intentar usar tabulate para un formato profesional
    if HAS_TABULATE:
        print(tabulate(data, headers=headers, tablefmt="grid"))
    else:
        # Fallback simple si tabulate no está instalado
        col_widths = [max(len(item) for item in col) for col in zip(*(data + [headers]))]
        row_format = " | ".join([f"{{:<{w}}}" for w in col_widths])
        print(row_format.format(*headers))
        print("-" * (sum(col_widths) + len(col_widths) * 3))
        for row in data:
            print(row_format.format(*row))
    print()


def run_tests():
    logger.info("=== INICIANDO VALIDACIÓN DE RECUPERACIÓN BASADA EN REQUISITOS (QA) ===")
    
    # 1. Inicializar recomendador con la base de datos Chroma de producción
    db_path = os.path.join(project_root, "chroma_db")
    if not os.path.exists(db_path):
        logger.error(f"No existe la base de datos ChromaDB en {db_path}.")
        logger.error("Por favor, ejecuta primero la indexación del proyecto (run_indexacion.py).")
        sys.exit(1)
        
    logger.info(f"Cargando base de datos ChromaDB desde: {db_path}...")
    recommender = OntologyRecommender(
        persist_directory=db_path,
        warmup=False
    )
    
    # Casos de prueba derivados directamente del dataset de requerimientos reales
    TEST_CASES = [
        {
            "id": 1,
            "requirement": "Zones are areas with spatial 3D volumes",
            "expected_ontology": "bot.ttl",
            "concept": "Zone",
            "semantic_query": "three-dimensional volume spatial extent"
        },
        {
            "id": 13,
            "requirement": "A device performs one or more functions",
            "expected_ontology": "saref_2020-05-29.n3",
            "concept": "device",
            "semantic_query": "hardware apparatus performing functions"
        },
        {
            "id": 35,
            "requirement": "It should be possible to define policies of type Assertion.",
            "expected_ontology": "odrl_2017-09-16.n3",
            "concept": "Assertion",
            "semantic_query": "policy statement rules and constraints"
        },
        {
            "id": 39,
            "requirement": "What is the genre of the game?",
            "expected_ontology": "GameOntologyv3.owl",
            "concept": "genre",
            "semantic_query": "game category type of video game style"
        },
        {
            "id": 55,
            "requirement": "What software can perform clustering task?",
            "expected_ontology": "swo.owl",
            "concept": "software",
            "semantic_query": "computer application algorithm program tools"
        }
    ]


    # ==========================================
    # VALIDACIÓN LÉXICA (BM25)
    # ==========================================
    logger.info("\n==========================================")
    logger.info(" 1. VALIDACIÓN LÉXICA (BM25)")
    logger.info("==========================================")
    
    for case in TEST_CASES:
        term = case["concept"]
        expected = case["expected_ontology"]
        logger.info(f"Req {case['id']}: Buscando término exacto '{term}' (Esperado: {expected})")
        
        docs = recommender._hybrid_retrieve(term, k=5, retrieval_mode="bm25")
        
        datos = []
        hit = False
        for idx, doc in enumerate(docs, start=1):
            ont = doc.metadata.get("source", "Desconocida")
            uri = doc.metadata.get("uri", "N/A")
            if expected.lower() in ont.lower():
                hit = True
                status_str = "ACERTADO"
            else:
                status_str = ""
            
            label = "N/A"
            for line in doc.page_content.split("\n"):
                if line.startswith("Label:"):
                    label = line.replace("Label:", "").strip()
                    break
            datos.append([idx, ont, label, uri[:60], status_str])
            
        print_table(
            headers=["#", "Ontología", "Label", "URI", "Estado"],
            data=datos,
            title=f"Búsqueda BM25: '{term}' (Caso {case['id']})"
        )
        
        if hit:
            logger.info(f"Éxito Léxico: La ontología '{expected}' se recuperó en el top-5.")
        else:
            logger.warning(f"Fallo Léxico: La ontología '{expected}' NO se encontró en el top-5.")

    # ==========================================
    # VALIDACIÓN SEMÁNTICA (Dense BGE-M3)
    # ==========================================
    logger.info("\n==========================================")
    logger.info(" 2. VALIDACIÓN SEMÁNTICA (Dense BGE-M3)")
    logger.info("==========================================")
    
    for case in TEST_CASES:
        query = case["semantic_query"]
        expected = case["expected_ontology"]
        logger.info(f"Req {case['id']}: Buscando semánticamente '{query}' (Esperado: {expected})")
        
        docs = recommender._hybrid_retrieve(query, k=5, retrieval_mode="dense")
        
        datos = []
        hit = False
        for idx, doc in enumerate(docs, start=1):
            ont = doc.metadata.get("source", "Desconocida")
            uri = doc.metadata.get("uri", "N/A")
            if expected.lower() in ont.lower():
                hit = True
                status_str = "ACERTADO"
            else:
                status_str = ""
                
            label = "N/A"
            for line in doc.page_content.split("\n"):
                if line.startswith("Label:"):
                    label = line.replace("Label:", "").strip()
                    break
            datos.append([idx, ont, label, uri[:60], status_str])
            
        print_table(
            headers=["#", "Ontología", "Label", "URI", "Estado"],
            data=datos,
            title=f"Búsqueda Vectorial: '{query}' (Caso {case['id']})"
        )
        
        if hit:
            logger.info(f"Éxito Semántico: La ontología '{expected}' se recuperó mediante sinónimos.")
        else:
            logger.warning(f"Fallo Semántico: La ontología '{expected}' NO se encontró mediante sinónimos.")

    # ==========================================
    # VALIDACIÓN DE REORDENAMIENTO (RERANKER)
    # ==========================================
    logger.info("\n==========================================")
    logger.info(" 3. IMPACTO DEL RERANKER (Cross-Encoder)")
    logger.info("==========================================")
    
    for case in TEST_CASES:
        req = case["requirement"]
        expected = case["expected_ontology"]
        logger.info(f"Req {case['id']}: Evaluando Reranker para '{req}' (Esperado: {expected})")
        
        raw_docs = recommender._hybrid_retrieve(req, k=10, retrieval_mode="hybrid")
        if not raw_docs:
            continue
            
        orden_previo = []
        for idx, doc in enumerate(raw_docs, start=1):
            ont = doc.metadata.get("source", "Desconocida")
            uri = doc.metadata.get("uri", "N/A")
            orden_previo.append({"doc": doc, "prev_rank": idx, "ontology": ont, "uri": uri})
            
        doc_contents = [d.page_content[:500] for d in raw_docs]
        pairs = [[req, content] for content in doc_contents]
        scores = recommender.reranker.predict(pairs)
        
        scored_docs = list(zip(raw_docs, scores))
        scored_docs_sorted = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        
        datos_comparacion = []
        rank_expected_pre = -1
        rank_expected_post = -1
        
        for rank_after, (doc, score) in enumerate(scored_docs_sorted, start=1):
            ontologia = doc.metadata.get("source", "Desconocida")
            uri = doc.metadata.get("uri", "N/A")
            
            rank_before = -1
            for item in orden_previo:
                if item["uri"] == uri:
                    rank_before = item["prev_rank"]
                    break
                    
            if expected.lower() in ontologia.lower():
                status_str = "ESPERADA"
                if rank_expected_post == -1 or rank_after < rank_expected_post:
                    rank_expected_post = rank_after
                    rank_expected_pre = rank_before
            else:
                status_str = ""
                
            desplazamiento = rank_before - rank_after
            if desplazamiento > 0:
                cambio = f"▲ +{desplazamiento}"
            elif desplazamiento < 0:
                cambio = f"▼ {desplazamiento}"
            else:
                cambio = "="
                
            datos_comparacion.append([
                rank_after,
                rank_before,
                cambio,
                f"{score:.4f}",
                ontologia,
                uri[:50],
                status_str
            ])
            
        print_table(
            headers=["Post-Rerank", "Pre-Rerank", "Desplazamiento", "Score Rerank", "Ontología", "URI", "Estado"],
            data=datos_comparacion,
            title=f"Reranking de Candidatos (Caso {case['id']})"
        )
        
        if rank_expected_post != -1:
            logger.info(f"Reranker hit: '{expected}' pasó de la posición {rank_expected_pre} a la {rank_expected_post}.")
        else:
            logger.warning(f"Reranker miss: '{expected}' no fue encontrada en el top-10 de candidatos.")
            
    logger.info("=== SUITE DE PRUEBAS DE RECUPERACIÓN COMPLETADA ===")


if __name__ == "__main__":
    run_tests()
