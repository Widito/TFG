import csv
import os  
import pandas as pd
from rag_basico import OntologyRecommender

# 1. Obtenemos la ruta de la carpeta donde está ESTE archivo (evaluate_rag.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Construimos las rutas relativas a este archivo, no a la consola
# El CSV está una carpeta arriba (..)
CSV_PATH = os.path.join(current_dir, "..", "dataset_bot_test.csv")
OUTPUT_CSV = os.path.join(current_dir, "..", "resultados_evaluacion.csv")

def evaluate():
    print("--- INICIANDO EVALUACIÓN AUTOMATIZADA ---")
    print(f"Leyendo dataset desde: {os.path.abspath(CSV_PATH)}") # Debug para que veas la ruta real
    
    # 1. Cargar el sistema
    rag = OntologyRecommender()
    
    # 2. Leer dataset
    try:
        # Usamos la ruta absoluta construida
        df = pd.read_csv(CSV_PATH, sep=";")
    except Exception as e:
        print(f"Error leyendo CSV: {e}")
        print(f"Verifica que el archivo existe en: {CSV_PATH}")
        return

    results = []
    
    print(f"Evaluando {len(df)} casos de prueba...\n")

    for index, row in df.iterrows():
        query = row['query_natural']
        target = row['expected_ontology']
        
        print(f"Prueba {index+1}/{len(df)}: '{query[:40]}...' -> Esperado: {target}")
        
        # Ejecutar RAG
        response = rag.run_pipeline(query, top_k=15)
        
        # --- CÁLCULO DE MÉTRICAS ---
        
        # 1. Retrieval Recall (¿Estaba el archivo en los 15 documentos?)
        retrieved_list = response['unique_retrieved_sources']
        hit_retrieval = target in retrieved_list
        
        # 2. Generación Accuracy (¿El LLM recomendó el archivo correcto?)
        # Buscamos el nombre del archivo en la respuesta de texto del LLM
        llm_text = response['llm_response'].lower()
        hit_generation = target.lower() in llm_text
        
        # Guardar métricas
        results.append({
            "id": row['id'],
            "query": query,
            "expected": target,
            "retrieved_sources": retrieved_list,
            "hit_retrieval": 1 if hit_retrieval else 0,
            "hit_generation": 1 if hit_generation else 0,
            "llm_output_snippet": response['llm_response'][:100].replace('\n', ' ')
        })

    # 3. Guardar y Mostrar Resumen
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False, sep=";")
    
    print("\n" + "="*40)
    print("RESULTADOS FINALES")
    print("="*40)
    print(f"Total pruebas: {len(df)}")
    print(f"Precisión Recuperación (MMR): {results_df['hit_retrieval'].mean()*100:.1f}%")
    print(f"Precisión Generación (LLM):   {results_df['hit_generation'].mean()*100:.1f}%")
    print(f"\nDetalle guardado en: {OUTPUT_CSV}")

if __name__ == "__main__":
    evaluate()