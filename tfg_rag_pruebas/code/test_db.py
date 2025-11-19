# test_db.py
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Rutas (ajusta si tu carpeta se llama diferente)
persist_directory = "tfg_rag_pruebas/chroma_db"

print("--- VERIFICANDO BASE DE DATOS ---")

# 1. Cargar la DB
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

# 2. Sacar un documento cualquiera
print("Buscando un documento de prueba...")
docs = vectorstore.similarity_search("Class", k=1)

if docs:
    print("\n¡ÉXITO! Se encontró información.")
    print("-" * 20)
    print(f"Contenido (primeros 50 chars): {docs[0].page_content[:50]}...")
    print("-" * 20)
    print(f"METADATOS ENCONTRADOS: {docs[0].metadata}")
    print("-" * 20)
    
    if "source" in docs[0].metadata:
        print(" PRUEBA SUPERADA: El campo 'source' existe en los metadatos.")
        print(f"   Origen detectado: {docs[0].metadata['source']}")
    else:
        print(" ERROR: No veo el campo 'source' en los metadatos. Algo falló en la indexación.")
else:
    print(" ERROR: La base de datos parece estar vacía.")