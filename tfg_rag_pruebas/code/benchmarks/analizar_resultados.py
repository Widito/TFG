import os
import sys
import json
import csv
import logging
from pathlib import Path
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


# Configurar sys.path para importar localmente desde src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, os.path.join(project_root, "src"))

from ontology_rag.evaluador_requisitos import EvaluadorRequisitos

# Configurar logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("AnalizarResultados")


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
    if HAS_TABULATE:
        print(tabulate(data, headers=headers, tablefmt="grid"))
    else:
        # Fallback simple
        col_widths = [max(len(item) for item in col) for col in zip(*(data + [headers]))]
        row_format = " | ".join([f"{{:<{w}}}" for w in col_widths])
        print(row_format.format(*headers))
        print("-" * (sum(col_widths) + len(col_widths) * 3))
        for row in data:
            print(row_format.format(*row))
    print()



def buscar_archivo_trazas_mas_reciente():
    """Busca el archivo de trazas JSON más reciente en la carpeta de reportes o resultado."""
    directorios = [
        os.path.join(project_root, "reportes_benchmarking"),
        os.path.join(project_root, "resultado")
    ]
    
    candidatos = []
    for d in directorios:
        if os.path.exists(d):
            for file in os.listdir(d):
                if file.endswith(".json") and (file.startswith("trazas_") or file == "trazas_ejecucion.json"):
                    filepath = os.path.join(d, file)
                    candidatos.append((filepath, os.path.getmtime(filepath)))
                    
    if not candidatos:
        return None
        
    # Ordenar por fecha de modificación descendente
    candidatos.sort(key=lambda x: x[1], reverse=True)
    return candidatos[0][0]


def leer_dataset_referencia():
    """Lee el dataset original dataset_bot_test.csv para alinear metadatos si es necesario."""
    csv_path = os.path.join(project_root, "dataset_bot_test.csv")
    if not os.path.exists(csv_path):
        return {}
        
    mapping = {}
    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            # Detectar delimitador
            sample = f.read(1024)
            f.seek(0)
            delimiter = ';' if ';' in sample else ','
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                q_raw = row.get('query_natural', '')
                q_norm = EvaluadorRequisitos._normalizar_requisito(q_raw)
                mapping[q_norm] = {
                    "id": row.get('id', ''),
                    "query": q_raw,
                    "expected": row.get('expected_ontology', '').strip(),
                    "difficulty": row.get('difficulty', 'unknown').strip()
                }
    except Exception as e:
        logger.warning(f"No se pudo cargar el dataset de referencia {csv_path}: {e}")
        
    return mapping


def analizar_formato_benchmark(data):
    """Analiza las trazas generadas por run_advanced_benchmark.py (lista de configuraciones)."""
    print("\n=== FORMATO DETECTADO: GRID SEARCH BENCHMARK ===")
    
    # 1. Mostrar configuraciones disponibles
    print("Configuraciones disponibles en este reporte:")
    for idx, run in enumerate(data, start=1):
        print(f"  {idx}. {run.get('config_name')} (Retrieval: {run.get('retrieval_mode')}, Rerank: {run.get('use_reranker')})")
        
    # Seleccionar la primera configuración por defecto (o preguntar si fuera interactivo)
    config_idx = 0
    run = data[config_idx]
    logger.info(f"Analizando resultados para: '{run.get('config_name')}'")
    
    cases = run.get("cases", [])
    if not cases:
        logger.error("No se encontraron casos de prueba en la configuración elegida.")
        return
        
    # 2. Calcular métricas agregadas
    total = len(cases)
    retrieval_hits = sum(1 for c in cases if c.get("hits", {}).get("retrieval") == 1)
    judge_hits = sum(1 for c in cases if c.get("hits", {}).get("judge") == 1)
    veredict_hits = sum(1 for c in cases if c.get("hits", {}).get("veredict") == 1)
    
    filter_recall = (retrieval_hits / total) * 100 if total > 0 else 0
    final_choice_accuracy = (judge_hits / total) * 100 if total > 0 else 0
    veredict_accuracy = (veredict_hits / total) * 100 if total > 0 else 0
    
    print("\n" + "="*40)
    print(" MÉTRICAS AGREGADAS FINALES")
    print("="*40)
    print(f"Total de Requisitos Evaluados: {total}")
    print(f"Filter Recall (Recuperación):  {filter_recall:.2f}%")
    print(f"Final Choice Accuracy (Juez):   {final_choice_accuracy:.2f}%")
    print(f"Veredict Accuracy (Recomendación): {veredict_accuracy:.2f}%")
    
    # 3. Métricas agrupadas por dificultad
    by_diff = {}
    for c in cases:
        diff = c.get("difficulty", "unknown").lower()
        if diff not in by_diff:
            by_diff[diff] = []
        by_diff[diff].append(c)
        
    datos_dificultad = []
    for diff in sorted(by_diff.keys()):
        diff_cases = by_diff[diff]
        d_total = len(diff_cases)
        d_ret = sum(1 for c in diff_cases if c.get("hits", {}).get("retrieval") == 1)
        d_judge = sum(1 for c in diff_cases if c.get("hits", {}).get("judge") == 1)
        d_ver = sum(1 for c in diff_cases if c.get("hits", {}).get("veredict") == 1)
        
        d_rec = (d_ret / d_total) * 100 if d_total > 0 else 0
        d_acc = (d_judge / d_total) * 100 if d_total > 0 else 0
        d_vac = (d_ver / d_total) * 100 if d_total > 0 else 0
        
        datos_dificultad.append([
            diff.capitalize(),
            d_total,
            f"{d_rec:.1f}%",
            f"{d_acc:.1f}%",
            f"{d_vac:.1f}%"
        ])
        
    print_table(
        headers=["Dificultad", "Casos", "Filter Recall", "Final Choice Acc", "Veredict Acc"],
        data=datos_dificultad,
        title="Rendimiento Agrupado por Dificultad"
    )
    
    # 4. Extraer ejemplos cualitativos (2 éxitos, 2 fallos)
    exitantes = [c for c in cases if c.get("status") == "SUCCESS"]
    if not exitantes:
        exitantes = [c for c in cases if c.get("hits", {}).get("judge") == 1]
        logger.info("(No se detectaron éxitos de veredicto final, mostrando aciertos a nivel de Juez LLM)")
    fallidos = [c for c in cases if c.get("hits", {}).get("judge") == 0]
    
    mostrar_ejemplos_cualitativos(exitantes[:2], fallidos[:2])



def analizar_formato_estandar(data):
    """Analiza las trazas generadas por la orquestación directa (tad_requisitos dictionary)."""
    print("\n=== FORMATO DETECTADO: TRAZAS DE EVALUACIÓN ESTÁNDAR ===")
    
    # Cargar metadatos del dataset original
    dataset_map = leer_dataset_referencia()
    if not dataset_map:
        logger.error("No se pudo cargar el dataset de referencia dataset_bot_test.csv.")
        logger.error("Se requiere este archivo para mapear las ontologías esperadas y dificultades.")
        return
        
    tad_reqs = data.get("tad_requisitos", {})
    if not tad_reqs:
        logger.error("No se encontró la clave 'tad_requisitos' en el archivo JSON.")
        return
        
    # Alinear y construir casos
    cases = []
    for query_text, trace in tad_reqs.items():
        q_norm = EvaluadorRequisitos._normalizar_requisito(query_text)
        meta = dataset_map.get(q_norm)
        
        if not meta:
            # Buscar coincidencia parcial si la normalización difiere levemente
            meta = next((v for k, v in dataset_map.items() if q_norm in k or k in q_norm), None)
            
        if not meta:
            # Caso no mapeado en el CSV
            continue
            
        expected = meta["expected"]
        difficulty = meta["difficulty"]
        
        # Evaluar hits
        raw_retrieved = trace.get("raw_retrieved_ontologies", [])
        retrieval_hit = any(expected.lower() in r.lower() for r in raw_retrieved)
        
        entidades_aprobadas = trace.get("entidades_aprobadas", [])
        approved_sources = list(set([ent.get("ontologia", "") for ent in entidades_aprobadas]))
        judge_hit = any(expected.lower() in a.lower() for a in approved_sources)
        
        cases.append({
            "id": meta["id"],
            "query": query_text,
            "expected": expected,
            "difficulty": difficulty,
            "status": "SUCCESS" if judge_hit else "FAIL_JUDGE" if retrieval_hit else "FAIL_RETRIEVAL",
            "hits": {
                "retrieval": 1 if retrieval_hit else 0,
                "judge": 1 if judge_hit else 0
            },
            "details": {
                "keywords": trace.get("keywords_usadas", ""),
                "raw_retrieved": raw_retrieved,
                "approved_sources": approved_sources,
                "entidades_aprobadas": entidades_aprobadas
            }
        })
        
    # Calcular métricas
    total = len(cases)
    retrieval_hits = sum(c["hits"]["retrieval"] for c in cases)
    judge_hits = sum(c["hits"]["judge"] for c in cases)
    
    filter_recall = (retrieval_hits / total) * 100 if total > 0 else 0
    final_choice_accuracy = (judge_hits / total) * 100 if total > 0 else 0
    
    print("\n" + "="*40)
    print(" MÉTRICAS AGREGADAS FINALES (ALINEADO CON CSV)")
    print("="*40)
    print(f"Total de Requisitos Evaluados: {total}")
    print(f"Filter Recall (Recuperación):  {filter_recall:.2f}%")
    print(f"Final Choice Accuracy (Juez):   {final_choice_accuracy:.2f}%")
    
    # Agrupar por dificultad
    by_diff = {}
    for c in cases:
        diff = c.get("difficulty", "unknown").lower()
        if diff not in by_diff:
            by_diff[diff] = []
        by_diff[diff].append(c)
        
    datos_dificultad = []
    for diff in sorted(by_diff.keys()):
        diff_cases = by_diff[diff]
        d_total = len(diff_cases)
        d_ret = sum(c["hits"]["retrieval"] for c in diff_cases)
        d_judge = sum(c["hits"]["judge"] for c in diff_cases)
        
        d_rec = (d_ret / d_total) * 100 if d_total > 0 else 0
        d_acc = (d_judge / d_total) * 100 if d_total > 0 else 0
        
        datos_dificultad.append([
            diff.capitalize(),
            d_total,
            f"{d_rec:.1f}%",
            f"{d_acc:.1f}%"
        ])
        
    print_table(
        headers=["Dificultad", "Casos", "Filter Recall", "Final Choice Acc"],
        data=datos_dificultad,
        title="Rendimiento Agrupado por Dificultad"
    )
    
    # 4. Ejemplos
    exitantes = [c for c in cases if c.get("status") == "SUCCESS"]
    if not exitantes:
        exitantes = [c for c in cases if c.get("hits", {}).get("judge") == 1]
        logger.info("(No se detectaron éxitos de veredicto final, mostrando aciertos a nivel de Juez LLM)")
    fallidos = [c for c in cases if c.get("hits", {}).get("judge") == 0]
    
    mostrar_ejemplos_cualitativos(exitantes[:2], fallidos[:2])


def mostrar_ejemplos_cualitativos(exitantes, fallidos):
    """Muestra ejemplos detallados de éxitos y fallos."""
    print("\n" + "="*50)
    print(" CASOS DE ESTUDIO CUALITATIVOS (EVIDENCIAS DE QA)")
    print("="*50)
    
    print("\n>>> EJEMPLOS EXITOSOS (SUCCESS/APPROVED) <<<")
    if not exitantes:
        print("  No hay casos exitosos o aprobados en este reporte.")
    for idx, c in enumerate(exitantes, start=1):
        print(f"\nCaso de Éxito {idx}: ID {c.get('id')} [{c.get('difficulty').upper()}]")
        print(f"  - Requisito: \"{c.get('query')}\"")
        print(f"  - Ontología Esperada: {c.get('expected')}")
        
        detalles = c.get("details", {})
        print(f"  - Palabras Clave RAG: {detalles.get('keywords', '').strip().replace('\n', ' | ')[:150]}...")
        print(f"  - Fuentes Recuperadas (Inicial): {detalles.get('raw_retrieved', [])[:5]}")
        
        # Entidades aprobadas
        ents = detalles.get("entidades_aprobadas", [])
        if not ents:
            # En formato benchmark de grid search las entidades aprobadas se mapean en details
            ents = detalles.get("approved_uris", [])
        
        print(f"  - Entidades Aprobadas por el Juez LLM (Evidencia Semántica):")
        for ent in ents[:2]:
            if isinstance(ent, dict):
                print(f"    * URI: {ent.get('uri')}")
                print(f"      Fragmento: {ent.get('texto', '')[:140].replace('\n', ' ')}...")
            else:
                print(f"    * URI: {ent}")
                
    print("\n>>> EJEMPLOS FALLIDOS (FAILURES) <<<")
    if not fallidos:
        print("  No hay casos fallidos en este reporte.")
    for idx, c in enumerate(fallidos, start=1):
        print(f"\nCaso de Fallo {idx}: ID {c.get('id')} [{c.get('difficulty').upper()}]")
        print(f"  - Requisito: \"{c.get('query')}\"")
        print(f"  - Ontología Esperada: {c.get('expected')}")
        print(f"  - Diagnóstico QA: {c.get('status')}")
        
        detalles = c.get("details", {})
        print(f"  - Fuentes Recuperadas (Inicial): {detalles.get('raw_retrieved', [])[:5]}")
        
        # Qué aprobó el juez
        approved = detalles.get("approved_sources", [])
        print(f"  - Fuentes Aprobadas por el Juez: {approved if approved else 'Ninguna'}")


def main():
    target_file = None
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
        
    if not target_file:
        target_file = buscar_archivo_trazas_mas_reciente()
        
    if not target_file or not os.path.exists(target_file):
        logger.error("No se encontró ningún archivo de trazas JSON para analizar.")
        logger.error("Por favor ejecuta primero un benchmark o especifica la ruta de un JSON como parámetro:")
        logger.error("Ejemplo: python analizar_resultados.py reportes_benchmarking/trazas_todos_los_requisitos_20260616_2025.json")
        sys.exit(1)
        
    logger.info(f"Procesando archivo de resultados: {os.path.abspath(target_file)}")
    
    with open(target_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except Exception as e:
            logger.error(f"Error parseando archivo JSON: {e}")
            sys.exit(1)
            
    # Detectar el formato
    if isinstance(data, list):
        # Formato de run_advanced_benchmark.py (Lista de configuraciones)
        analizar_formato_benchmark(data)
    elif isinstance(data, dict):
        # Formato estándar de trazas_ejecucion.json
        analizar_formato_estandar(data)
    else:
        logger.error("Formato de JSON desconocido. Debe ser una lista de ejecuciones o un diccionario de trazas.")


if __name__ == "__main__":
    main()
