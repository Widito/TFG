import csv
import pandas as pd
from rag_basico import OntologyRecommender

# Configuración
CSV_PATH = "../dataset_bot_test.csv" # Asegúrate de la ruta correcta
OUTPUT_CSV = "../resultados_evaluacion.csv"

def evaluate():
    print("--- INICIANDO EVALUACIÓN AUTOMATIZADA ---")
    
    # 1. Cargar el sistema
    rag = OntologyRecommender()
    
    # 2. Leer dataset
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