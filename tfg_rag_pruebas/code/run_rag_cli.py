import os

from ontology_rag import OntologyRecommender


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    persist_directory = os.path.join(project_root, "chroma_db")

    rag = OntologyRecommender(persist_directory=persist_directory)
    while True:
        q = input("\nConsulta ('salir'): ")
        if q == "salir":
            break
        try:
            res = rag.run_pipeline(q)
            print(f"Filtrados (Top-10 Reranked): {res['unique_retrieved_sources']}")
            print(f"Respuesta:\n{res['llm_response']}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
