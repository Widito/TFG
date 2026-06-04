import csv
import os  
import pandas as pd
# CORRECCIÓN 1: Importamos desde la librería instalable
from ontology_rag import OntologyRecommender
import sys
import time

class DualLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() 

    def flush(self):
        self.terminal.flush()
        self.log.flush()

current_dir = os.path.dirname(os.path.abspath(__file__))
# Calculamos rutas absolutas hacia la raíz del proyecto
project_root = os.path.dirname(current_dir)
CSV_PATH = os.path.join(project_root, "dataset_bot_test.csv")
OUTPUT_CSV = os.path.join(project_root, "resultados_evaluacion.csv")
# CORRECCIÓN 2: Calculamos la ruta de ChromaDB
PERSIST_DIRECTORY = os.path.join(project_root, "chroma_db")

log_filename = os.path.join(current_dir, "log_evaluacion_completa.txt")
sys.stdout = DualLogger(log_filename)

def evaluate():
    print("INICIANDO EVALUACIÓN AUTOMATIZADA ")
    print(f"Leyendo dataset desde: {os.path.abspath(CSV_PATH)}")
    
    # CORRECCIÓN 3: Inyectamos la dependencia (persist_directory)
    try:
        rag = OntologyRecommender(persist_directory=PERSIST_DIRECTORY)
    except Exception as e:
        print(f"Error iniciando RAG: {e}")
        return
    
    if not os.path.exists(CSV_PATH):
        print(f" ERROR: No se encuentra el dataset en: {CSV_PATH}")
        return

    try:
        df = pd.read_csv(CSV_PATH, sep=";")
    except Exception as e:
        print(f"Error leyendo CSV: {e}")
        return

    results = []
    
    print(f"Evaluando {len(df)} casos de prueba...\n")

    for index, row in df.iterrows():
        query = row['query_natural']
        target = row['expected_ontology']
        
        print(f"Prueba {index+1}/{len(df)}: '{query[:40]}...' -> Esperado: {target}")
        
        try:
            response = rag.run_pipeline(query, initial_k=100)
            
            retrieved_list = response.get('unique_retrieved_sources', [])
            hit_retrieval = target in retrieved_list
            
            llm_text = response.get('llm_response', '').lower()
            hit_generation = target.lower() in llm_text
            
            results.append({
                "id": row['id'],
                "query": query,
                "expected": target,
                "retrieved_sources": retrieved_list,
                "hit_retrieval": 1 if hit_retrieval else 0,
                "hit_generation": 1 if hit_generation else 0,
                "llm_output_snippet": response.get('llm_response', '')[:100].replace('\n', ' ')
            })
            
        except Exception as e:
            print(f"Error en prueba {index+1}: {e}")
            results.append({
                "id": row['id'],
                "query": query,
                "expected": target,
                "retrieved_sources": [],
                "hit_retrieval": 0,
                "hit_generation": 0,
                "llm_output_snippet": f"ERROR: {str(e)}"
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False, sep=";")
    
    print("\n" + "="*40)
    print("RESULTADOS FINALES (POST RE-RANKING)")
    print("="*40)
    if not results_df.empty:
        print(f"Total pruebas: {len(df)}")
        print(f"Precisión Recuperación (Filter Recall): {results_df['hit_retrieval'].mean()*100:.1f}%")
        print(f"Precisión Generación (Final Choice):    {results_df['hit_generation'].mean()*100:.1f}%")
        print(f"\nDetalle guardado en: {OUTPUT_CSV}")
    else:
        print("No se generaron resultados.")

if __name__ == "__main__":
    evaluate()