import os
import csv
import json
import time
import random
import shutil
import re
from datetime import datetime

from ontology_rag import EvaluadorRequisitos

def leer_dataset(ruta_csv):
    filas = []
    with open(ruta_csv, 'r', encoding='utf-8-sig') as f:
        # Intentar detectar delimitador
        sample = f.read(2048)
        f.seek(0)
        delimiter = ';' if ';' in sample else ','
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            filas.append(row)
    return filas

def extraer_ontologias_recomendadas(veredicto_texto, pool_ontologias):
    if not veredicto_texto:
        return []
    recomendadas = []
    text_lower = veredicto_texto.lower()
    for ont in pool_ontologias:
        prefix = ont.split('.')[0] if '.' in ont else ont
        pattern_full = re.compile(rf"\b{re.escape(ont.lower())}\b")
        pattern_prefix = re.compile(rf"\b{re.escape(prefix.lower())}\b")
        if pattern_full.search(text_lower) or pattern_prefix.search(text_lower):
            if ont not in recomendadas:
                recomendadas.append(ont)
    return recomendadas

def analizar_caso(fila, trace_requisito, recomendadas):
    expected = fila.get('expected_ontology', '').strip()
    
    # 1. Recuperación inicial (raw_retrieved_ontologies)
    raw_retrieved = trace_requisito.get('raw_retrieved_ontologies', [])
    retrieval_hit = any(expected.lower() in r.lower() for r in raw_retrieved)
    
    # 2. Re-ranked (entidades_recuperadas)
    entidades_recuperadas = trace_requisito.get('entidades_recuperadas', [])
    top_k_sources = list(set([ent.get('ontologia', '') for ent in entidades_recuperadas]))
    reranker_hit = any(expected.lower() in s.lower() for s in top_k_sources)
    
    # 3. Aprobadas por LLM Judge (entidades_aprobadas)
    entidades_aprobadas = trace_requisito.get('entidades_aprobadas', [])
    approved_sources = list(set([ent.get('ontologia', '') for ent in entidades_aprobadas]))
    judge_hit = any(expected.lower() in a.lower() for a in approved_sources)
    
    # 4. Recomendadas en veredicto (requiere aprobación del juez para este caso y recomendación de la ontología en el veredicto del lote)
    veredict_hit_batch = any(expected.lower() in rec.lower() for rec in recomendadas)
    veredict_hit = 1 if (judge_hit and veredict_hit_batch) else 0
    
    # Clasificación del estado / fallo
    if not retrieval_hit:
        status = "FAIL_RETRIEVAL"
    elif not reranker_hit:
        status = "FAIL_RERANKER"
    elif not judge_hit:
        status = "FAIL_JUDGE"
    elif not veredict_hit:
        status = "FAIL_VEREDICT_EXCLUSION"
    else:
        status = "SUCCESS"
        
    return {
        "id": fila.get('id', ''),
        "query": fila.get('query_natural', ''),
        "expected": expected,
        "difficulty": fila.get('difficulty', 'unknown'),
        "status": status,
        "hits": {
            "retrieval": 1 if retrieval_hit else 0,
            "reranker": 1 if reranker_hit else 0,
            "judge": 1 if judge_hit else 0,
            "veredict": 1 if veredict_hit else 0
        },
        "details": {
            "keywords": trace_requisito.get('keywords_usadas', ''),
            "raw_retrieved": raw_retrieved,
            "top_k_sources": top_k_sources,
            "approved_sources": approved_sources,
            "approved_uris": [ent.get('uri', '') for ent in entidades_aprobadas]
        }
    }

def generar_dashboard_html(nombre_tanda, config_runs, output_dir):
    fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fecha_hora_file = datetime.now().strftime("%Y%m%d_%H%M")
    nombre_archivo = f"dashboard_{nombre_tanda.replace(' ', '_').lower()}_{fecha_hora_file}.html"
    ruta_archivo = os.path.join(output_dir, nombre_archivo)
    
    # Serializar datos a JSON para inyectar en el JS del HTML
    datos_json = json.dumps(config_runs, indent=2, ensure_ascii=False)
    
    # Cargar plantilla HTML
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ruta_template = os.path.join(os.path.dirname(current_dir), "templates", "dashboard_template.html")
    if not os.path.exists(ruta_template):
        print(f"[ERROR] No se encuentra la plantilla HTML en {ruta_template}")
        return
        
    with open(ruta_template, 'r', encoding='utf-8') as f:
        html_template = f.read()
        
    html_content = html_template.replace("__NOMBRE_TANDA__", nombre_tanda)
    html_content = html_content.replace("__FECHA_HORA__", fecha_hora)
    html_content = html_content.replace("__DATOS_JSON__", datos_json)
    
    with open(ruta_archivo, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"\n[INFO] Dashboard HTML interactivo generado exitosamente en: {ruta_archivo}")

def ejecutar_benchmark_config(config_name, retrieval_mode, use_reranker, filas_tanda, ruta_csv, project_root):
    persist_directory = os.path.join(project_root, "chroma_db")
    output_dir = os.path.join(project_root, "resultado")
    evaluador = EvaluadorRequisitos(persist_directory=persist_directory, output_dir=output_dir)
    
    # Identificar la ruta de las trazas habituales para protegerlas
    trazas_path = os.path.join(output_dir, "trazas_ejecucion.json")
    backup_path = trazas_path + ".bak"
    trazas_existentes = False
    if os.path.exists(trazas_path):
        shutil.copy2(trazas_path, backup_path)
        trazas_existentes = True
        
    ids_seleccionados = [int(f['id']) for f in filas_tanda if 'id' in f and f['id'].strip().isdigit()]
    
    try:
        print(f"\n=== Ejecutando RAG: {config_name} ({len(filas_tanda)} requisitos) ===")
        start_time = time.time()
        
        # Ejecutar la evaluación para los IDs seleccionados con la config correspondiente
        resultados = evaluador.orquestar_evaluacion(
            ruta_csv=ruta_csv,
            selected_ids=ids_seleccionados,
            retrieval_mode=retrieval_mode,
            use_reranker=use_reranker
        )
        
        total_execution_time = time.time() - start_time
        
        # Cargar trazas detalladas recién generadas
        with open(trazas_path, 'r', encoding='utf-8') as f:
            trazas_json = json.load(f)
            
        tad_detallado = trazas_json.get("tad_requisitos", {})
        veredicto_final = resultados.get("veredicto_final", "")
        veredicto_estructurado = resultados.get("veredicto_estructurado", {})
        recomendadas = veredicto_estructurado.get("recomendadas", [])
        
        # Obtener pool de ontologías esperadas (únicas y no vacías)
        expected_set = set([f.get('expected_ontology', '').strip() for f in filas_tanda if f.get('expected_ontology', '').strip()])
        rec_set = set([r.strip() for r in recomendadas if r.strip()])
        
        # Calcular Precision, Recall y F1-Score de Red (Tanda)
        intersection = {r for r in rec_set if any(r.lower() == e.lower() for e in expected_set)}
        veredict_precision = len(intersection) / len(rec_set) if len(rec_set) > 0 else 0.0
        veredict_recall = len(intersection) / len(expected_set) if len(expected_set) > 0 else 0.0
        if veredict_precision + veredict_recall > 0:
            veredict_f1 = 2 * (veredict_precision * veredict_recall) / (veredict_precision + veredict_recall)
        else:
            veredict_f1 = 0.0
            
        # Analizar cada caso individualmente
        casos_analizados = []
        tiempos = []
        for fila in filas_tanda:
            query_raw = fila.get('query_natural', '')
            query_norm = EvaluadorRequisitos._normalizar_requisito(query_raw)
            trace_req = tad_detallado.get(query_norm)
            if trace_req is None:
                trace_req = tad_detallado.get(query_raw, {})
            # Estimar tiempo por consulta
            tiempos.append(total_execution_time / len(filas_tanda))
            caso = analizar_caso(fila, trace_req, recomendadas)
            casos_analizados.append(caso)
            
        # Calcular métricas globales
        total = len(casos_analizados)
        retrieval_hits = sum(c["hits"]["retrieval"] for c in casos_analizados)
        judge_hits = sum(c["hits"]["judge"] for c in casos_analizados)
        veredict_hits = sum(c["hits"]["veredict"] for c in casos_analizados)
        
        metrics = {
            "retrieval_recall": retrieval_hits / total if total > 0 else 0.0,
            "judge_accuracy": judge_hits / total if total > 0 else 0.0,
            "veredict_accuracy": veredict_hits / total if total > 0 else 0.0,
            "veredict_precision": veredict_precision,
            "veredict_recall": veredict_recall,
            "veredict_f1": veredict_f1,
            "average_time": total_execution_time / total if total > 0 else 0.0,
            "total_time": total_execution_time
        }
        
        print("\n--- Métricas Resultantes ---")
        print(f"Recall Recuperación Inicial:  {metrics['retrieval_recall']*100:.1f}%")
        print(f"Precisión Juez LLM:           {metrics['judge_accuracy']*100:.1f}%")
        print(f"Acierto Veredicto Individual: {metrics['veredict_accuracy']*100:.1f}%")
        print(f"Precisión Red Veredicto:      {metrics['veredict_precision']*100:.1f}%")
        print(f"Recall Red Veredicto:         {metrics['veredict_recall']*100:.1f}%")
        print(f"F1-Score Red Veredicto:       {metrics['veredict_f1']*100:.1f}%")
        print(f"Tiempo Medio:                 {metrics['average_time']:.2f}s/query")
        
        return {
            "config_name": config_name,
            "retrieval_mode": retrieval_mode,
            "use_reranker": use_reranker,
            "metrics": metrics,
            "cases": casos_analizados,
            "veredicto_final": veredicto_final,
            "veredicto_estructurado": veredicto_estructurado
        }
        
    finally:
        # Restaurar trazas originales
        if trazas_existentes:
            shutil.move(backup_path, trazas_path)
        elif os.path.exists(trazas_path):
            os.remove(trazas_path)

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    ruta_csv = os.path.join(project_root, "dataset_bot_test.csv")
    
    if not os.path.exists(ruta_csv):
        print(f"[ERROR] No se encuentra el dataset en {ruta_csv}")
        return
        
    filas = leer_dataset(ruta_csv)
    
    while True:
        print("\n" + "="*60)
        print("   SUITE DE BENCHMARKING AVANZADO - RAG ONTOLOGÍAS")
        print("="*60)
        print("1. Ejecutar Evaluación de Configuración Única (Híbrido + Reranker)")
        print("2. Ejecutar Grid Search Comparativo (Las 6 Configuraciones RAG)")
        print("3. Salir")
        
        opcion = input("Selecciona una opción (1-3): ").strip()
        
        if opcion not in ("1", "2"):
            if opcion == "3":
                print("Saliendo de la suite de benchmarking...")
                break
            else:
                print("Opción inválida.")
                continue
                
        # Submenú para seleccionar el subconjunto de datos
        print("\n--- Selecciona el Subconjunto de Datos ---")
        print("1. Todos los requisitos (36 casos)")
        print("2. Solo dificultad 'easy' (13 casos)")
        print("3. Solo dificultad 'medium' (13 casos)")
        print("4. Solo dificultad 'hard' (10 casos)")
        print("5. Muestra aleatoria mezclada (Mixed)")
        print("6. Requisitos específicos por ID (manual)")
        print("7. Requisitos de una ontología específica")
        
        op_subset = input("Selecciona una opción (1-7): ").strip()
        
        filas_filtradas = []
        tanda_name = ""
        
        if op_subset == "1":
            filas_filtradas = filas
            tanda_name = "Todos los Requisitos"
        elif op_subset == "2":
            filas_filtradas = [f for f in filas if f.get('difficulty') == 'easy']
            tanda_name = "Dificultad Easy"
        elif op_subset == "3":
            filas_filtradas = [f for f in filas if f.get('difficulty') == 'medium']
            tanda_name = "Dificultad Medium"
        elif op_subset == "4":
            filas_filtradas = [f for f in filas if f.get('difficulty') == 'hard']
            tanda_name = "Dificultad Hard"
        elif op_subset == "5":
            cantidad = input("Introduce la cantidad de casos a mezclar de forma aleatoria: ").strip()
            num = 5
            if cantidad.isdigit() and int(cantidad) > 0:
                num = int(cantidad)
            filas_filtradas = random.sample(filas, min(num, len(filas)))
            tanda_name = f"Muestra Aleatoria {len(filas_filtradas)} Reqs"
        elif op_subset == "6":
            ids_str = input("Introduce los IDs separados por coma (ej: 1,5,12): ").strip()
            try:
                ids_manuales = [int(x.strip()) for x in ids_str.split(',') if x.strip().isdigit()]
                filas_filtradas = [f for f in filas if 'id' in f and f['id'].strip().isdigit() and int(f['id']) in ids_manuales]
                tanda_name = f"Custom {len(filas_filtradas)} Reqs"
            except Exception as e:
                print(f"Error procesando IDs: {e}")
                continue
        elif op_subset == "7":
            # Obtener la lista única de ontologías esperadas del dataset
            ontologias_disponibles = sorted(list(set([f.get('expected_ontology', '').strip() for f in filas if f.get('expected_ontology', '').strip()])))
            if not ontologias_disponibles:
                print("No se encontraron ontologías en la columna 'expected_ontology'.")
                continue
            
            print("\n--- Selecciona la Ontología ---")
            for idx, ont in enumerate(ontologias_disponibles, start=1):
                print(f"{idx}. {ont}")
            
            op_ont = input(f"Selecciona una opción (1-{len(ontologias_disponibles)}): ").strip()
            if op_ont.isdigit() and 1 <= int(op_ont) <= len(ontologias_disponibles):
                ont_elegida = ontologias_disponibles[int(op_ont) - 1]
                filas_filtradas = [f for f in filas if f.get('expected_ontology', '').strip().lower() == ont_elegida.lower()]
                tanda_name = f"Ontologia {ont_elegida}"
            else:
                print("Opción de ontología inválida.")
                continue
        else:
            print("Opción inválida.")
            continue
            
        if not filas_filtradas:
            print("No se seleccionó ningún requisito.")
            continue
            
        # Directorio para guardar el dashboard final
        output_dir = os.path.join(project_root, "reportes_benchmarking")
        os.makedirs(output_dir, exist_ok=True)
        
        config_runs = []
        
        if opcion == "1":
            # Ejecutar una sola tanda: Híbrido + Reranker
            run_result = ejecutar_benchmark_config(
                config_name="Híbrido + Reranker (Línea Base)",
                retrieval_mode="hybrid",
                use_reranker=True,
                filas_tanda=filas_filtradas,
                ruta_csv=ruta_csv,
                project_root=project_root
            )
            config_runs.append(run_result)
        else:
            # Ejecutar Grid Search: las 6 combinaciones
            configs_to_run = [
                {"name": "Híbrido + Reranker", "mode": "hybrid", "rerank": True},
                {"name": "Híbrido Sin Reranker", "mode": "hybrid", "rerank": False},
                {"name": "Denso + Reranker", "mode": "dense", "rerank": True},
                {"name": "Denso Sin Reranker", "mode": "dense", "rerank": False},
                {"name": "BM25 + Reranker", "mode": "bm25", "rerank": True},
                {"name": "BM25 Sin Reranker", "mode": "bm25", "rerank": False},
            ]
            
            print(f"\n[INFO] Iniciando Grid Search de {len(configs_to_run)} configuraciones...")
            for config in configs_to_run:
                run_result = ejecutar_benchmark_config(
                    config_name=config["name"],
                    retrieval_mode=config["mode"],
                    use_reranker=config["rerank"],
                    filas_tanda=filas_filtradas,
                    ruta_csv=ruta_csv,
                    project_root=project_root
                )
                config_runs.append(run_result)
        
        # Generar el Dashboard HTML y las trazas JSON
        generar_dashboard_html(tanda_name, config_runs, output_dir)
        
        fecha_hora_file = datetime.now().strftime("%Y%m%d_%H%M")
        nombre_json = f"trazas_{tanda_name.replace(' ', '_').lower()}_{fecha_hora_file}.json"
        ruta_json = os.path.join(output_dir, nombre_json)
        with open(ruta_json, 'w', encoding='utf-8') as f:
            json.dump(config_runs, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Trazas detalladas del benchmark guardadas en: {ruta_json}")

if __name__ == "__main__":
    main()
