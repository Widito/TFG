import csv
import os  
import pandas as pd
from rag_basico import OntologyRecommender
import sys
import time

# --- CLASE PARA REDIRIGIR SALIDA (LOGGING) ---
class DualLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Asegura que se guarde en tiempo real

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirigir stdout al archivo y pantalla
log_filename = "log_evaluacion_completa.txt"
sys.stdout = DualLogger(log_filename)
# ---------------------------------------------


# 1. Obtenemos la ruta de la carpeta donde está ESTE archivo
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Construimos las rutas relativas (dataset_bot_test.csv está una carpeta arriba)
CSV_PATH = os.path.join(current_dir, "..", "dataset_bot_test.csv")
OUTPUT_CSV = os.path.join(current_dir, "..", "resultados_evaluacion.csv")

def evaluate():
    print("INICIANDO EVALUACIÓN AUTOMATIZADA ")
    print(f"Leyendo dataset desde: {os.path.abspath(CSV_PATH)}")
    
    # 1. Cargar el sistema
    try:
        rag = OntologyRecommender()
    except Exception as e:
        print(f"Error iniciando RAG: {e}")
        return
    
    # 2. Leer dataset
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
        
        # EJECUCIÓN DEL RAG
        # Usamos initial_k=40 para permitir que el Broad Retrieval capture candidatos antes de que el Re-ranker los filtre.
        try:
            response = rag.run_pipeline(query, initial_k=40)
            
            # CÁLCULO DE MÉTRICAS 
            
            # 1. Retrieval Recall (¿Sobrevivió el archivo correcto al re-rankeo?)
            # 'unique_retrieved_sources' ahora contiene la lista FILTRADA por el LLM
            retrieved_list = response.get('unique_retrieved_sources', [])
            hit_retrieval = target in retrieved_list
            
            # 2. Generación Accuracy (¿El LLM recomendó el archivo correcto?)
            llm_text = response.get('llm_response', '').lower()
            hit_generation = target.lower() in llm_text
            
            # Guardar métricas
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

    # 3. Guardar y Mostrar Resumen
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