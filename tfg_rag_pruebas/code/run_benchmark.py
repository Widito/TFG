import os
import csv
import json
import random
import shutil
from datetime import datetime

from ontology_rag import EvaluadorRequisitos

def leer_dataset(ruta_csv):
    filas = []
    with open(ruta_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            filas.append(row)
    return filas

def generar_reporte_markdown(nombre_tanda, filas_tanda, matriz_cobertura, veredicto, output_dir):
    fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fecha_hora_file = datetime.now().strftime("%Y%m%d_%H%M")
    
    nombre_archivo = f"reporte_{nombre_tanda.replace(' ', '_').lower()}_{fecha_hora_file}.md"
    ruta_archivo = os.path.join(output_dir, nombre_archivo)
    
    with open(ruta_archivo, 'w', encoding='utf-8') as f:
        f.write(f"# Reporte de Ejecución - {nombre_tanda}\n")
        f.write(f"- Fecha y Hora: {fecha_hora}\n")
        f.write(f"- Total de Requisitos Evaluados: {len(filas_tanda)}\n\n")
        
        f.write("## 1. Contexto de Entrada \n")
        f.write("| ID | Query Natural | Ontología Esperada |\n")
        f.write("|---|---|---|\n")
        for fila in filas_tanda:
            f.write(f"| {fila.get('id', '')} | {fila.get('query_natural', '')} | {fila.get('expected_ontology', '')} |\n")
        f.write("\n")
        
        f.write("## 2. Matriz de Cobertura Generada\n")
        f.write("```json\n")
        f.write(json.dumps(matriz_cobertura, indent=2, ensure_ascii=False) + "\n")
        f.write("```\n\n")
        
        f.write("## 3. Veredicto Final del LLM\n")
        f.write(veredicto + "\n")
        
    print(f"\n[INFO] Reporte generado exitosamente en: {ruta_archivo}")

def ejecutar_tanda(nombre_tanda, ids_seleccionados, filas_tanda, ruta_csv, project_root):
    # Inicializar el evaluador
    persist_directory = os.path.join(project_root, "chroma_db")
    evaluador = EvaluadorRequisitos(persist_directory=persist_directory)
    
    # Identificar la ruta de las trazas habituales para protegerlas
    # EvaluadorRequisitos guarda en: src/trazas_ejecucion.json
    trazas_path = os.path.join(project_root, "src", "trazas_ejecucion.json")
    
    # Backup trazas if exists
    backup_path = trazas_path + ".bak"
    trazas_existentes = False
    if os.path.exists(trazas_path):
        shutil.copy2(trazas_path, backup_path)
        trazas_existentes = True
        
    try:
        print(f"\n=== Iniciando {nombre_tanda} ===")
        # Ejecutar evaluación solo para los IDs seleccionados
        resultados = evaluador.orquestar_evaluacion(
            ruta_csv=ruta_csv,
            selected_ids=ids_seleccionados
        )
        
        # Crear la carpeta de reportes en la raíz del proyecto
        output_dir = os.path.join(project_root, "reportes_benchmarking")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generar el reporte Markdown
        generar_reporte_markdown(
            nombre_tanda=nombre_tanda,
            filas_tanda=filas_tanda,
            matriz_cobertura=resultados.get("matriz_cobertura", {}),
            veredicto=resultados.get("veredicto_final", ""),
            output_dir=output_dir
        )
        
        # Opcional: Guardar las trazas específicas de esta ejecución en la carpeta de reportes
        fecha_hora_file = datetime.now().strftime("%Y%m%d_%H%M")
        trazas_reporte = os.path.join(output_dir, f"trazas_{nombre_tanda.replace(' ', '_').lower()}_{fecha_hora_file}.json")
        if os.path.exists(trazas_path):
            shutil.copy2(trazas_path, trazas_reporte)
            
    finally:
        # Restaurar trazas originales
        if trazas_existentes:
            shutil.move(backup_path, trazas_path)
        elif os.path.exists(trazas_path):
            os.remove(trazas_path)

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    ruta_csv = os.path.join(project_root, "dataset_bot_test.csv")
    
    if not os.path.exists(ruta_csv):
        print(f"[ERROR] No se encuentra el dataset en {ruta_csv}")
        return
        
    filas = leer_dataset(ruta_csv)
    
    while True:
        print("\n" + "="*50)
        print("   MENÚ DE BENCHMARKING - RAG ONTOLOGÍAS")
        print("="*50)
        print("1. Tanda: Solo filas bot.ttl")
        print("2. Tanda: Solo filas saref_2020-05-29.n3")
        print("3. Tanda: Solo filas odrl_2017-09-16.n3")
        print("4. Tanda: Solo filas con dificultad 'hard'")
        print("5. Tanda: Solo filas con dificultad 'medium'")
        print("6. Tanda: Solo filas con dificultad 'easy'")
        print("7. Tanda: Muestra aleatoria mezclada (Mixed - 5 filas)")
        print("8. Tanda: Todos los requisitos")
        print("9. Tanda: Requisitos específicos por ID (manual)")
        print("10. Salir")
        
        opcion = input("Selecciona una opción (1-10): ")
        
        if opcion == "1":
            filas_filtradas = [f for f in filas if f.get('expected_ontology') == 'bot.ttl']
            ids = [int(f['id']) for f in filas_filtradas if 'id' in f and f['id'].strip().isdigit()]
            ejecutar_tanda("Tanda bot_ttl", ids, filas_filtradas, ruta_csv, project_root)
        elif opcion == "2":
            filas_filtradas = [f for f in filas if f.get('expected_ontology') == 'saref_2020-05-29.n3']
            ids = [int(f['id']) for f in filas_filtradas if 'id' in f and f['id'].strip().isdigit()]
            ejecutar_tanda("Tanda saref", ids, filas_filtradas, ruta_csv, project_root)
        elif opcion == "3":
            filas_filtradas = [f for f in filas if f.get('expected_ontology') == 'odrl_2017-09-16.n3']
            ids = [int(f['id']) for f in filas_filtradas if 'id' in f and f['id'].strip().isdigit()]
            ejecutar_tanda("Tanda odrl", ids, filas_filtradas, ruta_csv, project_root)
        elif opcion == "4":
            filas_filtradas = [f for f in filas if f.get('difficulty') == 'hard']
            ids = [int(f['id']) for f in filas_filtradas if 'id' in f and f['id'].strip().isdigit()]
            ejecutar_tanda("Tanda hard", ids, filas_filtradas, ruta_csv, project_root)
        elif opcion == "5":
            filas_filtradas = [f for f in filas if f.get('difficulty') == 'medium']
            ids = [int(f['id']) for f in filas_filtradas if 'id' in f and f['id'].strip().isdigit()]
            ejecutar_tanda("Tanda medium", ids, filas_filtradas, ruta_csv, project_root)
        elif opcion == "6":
            filas_filtradas = [f for f in filas if f.get('difficulty') == 'easy']
            ids = [int(f['id']) for f in filas_filtradas if 'id' in f and f['id'].strip().isdigit()]
            ejecutar_tanda("Tanda easy", ids, filas_filtradas, ruta_csv, project_root)
        elif opcion == "7":
            # Tomar muestra aleatoria de hasta 5 elementos
            muestra = random.sample(filas, min(5, len(filas)))
            ids = [int(f['id']) for f in muestra if 'id' in f and f['id'].strip().isdigit()]
            ejecutar_tanda("Tanda mixed", ids, muestra, ruta_csv, project_root)
        elif opcion == "8":
            ids = [int(f['id']) for f in filas if 'id' in f and f['id'].strip().isdigit()]
            ejecutar_tanda("Tanda todos", ids, filas, ruta_csv, project_root)
        elif opcion == "9":
            ids_str = input("Introduce los IDs o filas separados por coma (ej: 1, 5, 12): ")
            try:
                ids_manuales = [int(x.strip()) for x in ids_str.split(',') if x.strip().isdigit()]
                if not ids_manuales:
                    print("No se introdujeron IDs válidos.")
                    continue
                filas_filtradas = [f for f in filas if 'id' in f and f['id'].strip().isdigit() and int(f['id']) in ids_manuales]
                ejecutar_tanda(f"Tanda custom_{len(ids_manuales)}_reqs", ids_manuales, filas_filtradas, ruta_csv, project_root)
            except Exception as e:
                print(f"Error procesando IDs: {e}")
        elif opcion == "10":
            print("Saliendo...")
            break
        else:
            print("Opción inválida.")

if __name__ == "__main__":
    main()
