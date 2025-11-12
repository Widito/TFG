# (Versión DEBUG)
print("Iniciando el proceso de indexación (esto puede tardar varios minutos)...")

import rdflib
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import traceback # Para mostrar errores detallados

print("Paso 1: Librerías importadas.")
print("-" * 30)

# --- Define el directorio de la base de datos primero ---
persist_directory = "tfg_rag_pruebas/chroma_db"

# --- PASO 0: Limpieza (Opcional pero recomendado) ---
# Si el directorio ya existe, bórralo para empezar de cero.
# (Necesitas 'import shutil' al principio del script)
# try:
#     if os.path.exists(persist_directory):
#         print(f"Eliminando base de datos antigua en '{persist_directory}'...")
#         shutil.rmtree(persist_directory)
#         print("Base de datos antigua eliminada.")
# except Exception as e:
#     print(f"No se pudo eliminar el directorio antiguo: {e}")

try:
    # PASO 2: CARGAR TODAS LAS ONTOLOGÍAS
    print("Paso 2: Cargando todas las ontologías .n3 desde 'dataset'...")
    g = rdflib.Graph()
    ontologies_dir = "tfg_rag_pruebas/dataset" 
    files_loaded = 0

    if not os.path.exists(ontologies_dir):
        print(f"ERROR: El directorio 'dataset' no se encuentra en: {os.path.abspath(ontologies_dir)}")
        exit()

    for filename in os.listdir(ontologies_dir):
        if filename.endswith(".n3"):
            filepath = os.path.join(ontologies_dir, filename)
            try:
                print(f"  Cargando {filename}...")
                g.parse(filepath, format="n3")
                files_loaded += 1
            except Exception as e:
                print(f"    -> ERROR al parsear {filename}. Saltando archivo. Motivo: {e}")

    total_tripletas = len(g)
    if total_tripletas == 0:
        print("ADVERTENCIA: No se cargaron tripletas. ¿Está el directorio 'dataset' vacío o la ruta es incorrecta?")
        exit()
        
    print(f"Ontologías cargadas. El grafo total contiene {total_tripletas} tripletas.")
    print("-" * 30)

    # PASO 3: EXTRAER DOCUMENTOS CON SPARQL
    print("Paso 3: Ejecutando consulta SPARQL genérica...")
    query_text = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dc: <http://purl.org/dc/elements/1.1/>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX obo: <http://purl.obolibrary.org/obo/>

        SELECT ?subject ?label ?description
        WHERE {
            { ?subject rdfs:comment ?description . }
            UNION
            { ?subject dc:description ?description . }
            UNION
            { ?subject dcterms:description ?description . }
            UNION
            { ?subject skos:definition ?description . }
            UNION
            { ?subject obo:IAO_0000115 ?description . }
            
            FILTER(isLiteral(?description))
            
            OPTIONAL { ?subject rdfs:label ?label_rdfs . }
            OPTIONAL { ?subject dc:title ?label_dc . }
            OPTIONAL { ?subject skos:prefLabel ?label_skos . }
            
            BIND(COALESCE(?label_rdfs, ?label_dc, ?label_skos, "") AS ?label)
        }
    """
    results = g.query(query_text)

    documents = []
    for row in results:
        if len(row.description) > 5:
            doc_text = f"Concepto URI: {row.subject}\nEtiqueta: {str(row.label)}\nDescripción: {row.description}"
            documents.append(doc_text)

    if len(documents) == 0:
        print("ADVERTENCIA: La consulta SPARQL no devolvió ningún documento.")
        print("Esto podría deberse a que las ontologías no usan las propiedades de descripción esperadas.")
        exit()
        
    print(f"Consulta SPARQL completada. Se han extraído {len(documents)} descripciones.")
    print("-" * 30)

    # PASO 4: CONFIGURAR EMBEDDINGS Y CREAR DB
    print("Paso 4: Configurando embeddings y creando la base de datos persistente...")
    
    print("Cargando modelo de embeddings (BAAI/bge-m3)...")
    model_name = "BAAI/bge-m3"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print("Modelo de embeddings cargado.")

    print(f"Creando la base de datos vectorial en '{persist_directory}'...")
    print("Este es el paso lento. Por favor, espera...")

    # Crear la base de datos.
    vectorstore = Chroma.from_texts(
        texts=documents, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    
    # ¡Forzar el guardado en disco!
    vectorstore.persist()

    print("\n--- ¡ÉXITO! ---")
    print(f"Base de datos vectorial creada y guardada en '{persist_directory}'.")
    # Añadimos una comprobación final
    count = vectorstore._collection.count()
    print(f"Comprobación: La base de datos contiene {count} documentos indexados.")
    print("Ya puedes ejecutar '2_rag_persistente.py' para hacer consultas.")

except Exception as e:
    print("\n\n--- ERROR CRÍTICO DURANTE LA INDEXACIÓN ---")
    print(f"El script falló. Motivo: {e}")
    print("Detalles del error:")
    traceback.print_exc()