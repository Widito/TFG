import os
from ontology_rag import OntologyIndexer

def main():
    # Obtener rutas absolutas del proyecto
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    persist_directory = os.path.join(project_root, "chroma_db")

    # Definir los directorios origen de ontologías
    folders_to_process = [
        os.path.join(project_root, "dataset"),
        os.path.join(project_root, "gov_acad_dataset"),
        os.path.join(project_root, "dataset_noise_industry")
    ]

    print("Iniciando indexación de ontologías...")
    
    # Instanciar el indexador con Inyección de Dependencias
    indexer = OntologyIndexer(
        source_directories=folders_to_process,
        persist_directory=persist_directory,
        embedding_model="BAAI/bge-m3",
        batch_size=5000,
        recreate_db=True
    )
    
    # Ejecutar el pipeline de ingesta
    indexer.build_index()
    print("Indexación completada exitosamente")

if __name__ == "__main__":
    main()
