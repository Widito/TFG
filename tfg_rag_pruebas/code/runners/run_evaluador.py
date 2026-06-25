import os

from ontology_rag import EvaluadorRequisitos


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    persist_directory = os.path.join(project_root, "chroma_db")
    ruta_default = os.path.join(project_root, "dataset_bot_test.csv")

    evaluador = EvaluadorRequisitos(persist_directory=persist_directory)
    evaluador.orquestar_evaluacion(ruta_default, start_row=1, end_row=7)


if __name__ == "__main__":
    main()
