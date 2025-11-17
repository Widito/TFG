# (Versión DEBUG)
print("Iniciando el proceso de indexación")

import rdflib
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import traceback # Para mostrar errores detallados

print("Paso 1: Librerías importadas.")
print("-" * 30)

# --- Define el directorio de la base de datos primero ---
persist_directory = "tfg_rag_pruebas/chroma_db"

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
    print("Paso 3: Ejecutando consultas SPARQL para Clases y Propiedades...")
    
    # Query 1: Extraer Clases (rdfs:Class y owl:Class)
    query_classes = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        SELECT ?uri ?label ?comment
        WHERE {
            { ?uri a rdfs:Class . }
            UNION
            { ?uri a owl:Class . }
            
            FILTER(!isblank(?uri))

            OPTIONAL { ?uri rdfs:label ?label_rdfs . }
            OPTIONAL { ?uri rdfs:comment ?comment_rdfs . }
            
            # Coalesce para obtener al menos un label o el URI
            BIND(COALESCE(?label_rdfs, "") AS ?label)
            BIND(COALESCE(?comment_rdfs, "") AS ?comment)
        }
    """

    # Query 2: Extraer Propiedades (rdf:Property, owl:ObjectProperty, owl:DatatypeProperty)
    query_properties = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        SELECT ?uri ?label ?comment ?domain ?range
        WHERE {
            { ?uri a rdf:Property . }
            UNION
            { ?uri a owl:ObjectProperty . }
            UNION
            { ?uri a owl:DatatypeProperty . }
            
            FILTER(!isblank(?uri))

            OPTIONAL { ?uri rdfs:label ?label_rdfs . }
            OPTIONAL { ?uri rdfs:comment ?comment_rdfs . }
            OPTIONAL { ?uri rdfs:domain ?domain_rdfs . }
            OPTIONAL { ?uri rdfs:range ?range_rdfs . }
            
            BIND(COALESCE(?label_rdfs, "") AS ?label)
            BIND(COALESCE(?comment_rdfs, "") AS ?comment)
            BIND(COALESCE(STR(?domain_rdfs), "") AS ?domain)
            BIND(COALESCE(STR(?range_rdfs), "") AS ?range)
        }
    """

    documents = []

    # Procesar Clases
    print("Procesando Clases...")
    results_classes = g.query(query_classes)
    for row in results_classes:
        # Solo añadimos si tiene al menos un label o comentario
        if row.label or row.comment:
            doc_text = f"Tipo: Clase\nURI: {row.uri}\nEtiqueta: {row.label}\nDescripción: {row.comment}"
            documents.append(doc_text)

    print(f"  ... {len(documents)} documentos de Clases creados.")

    # Procesar Propiedades
    print("Procesando Propiedades...")
    results_properties = g.query(query_properties)
    count_props = 0
    for row in results_properties:
        # Solo añadimos si tiene al menos un label o comentario
        if row.label or row.comment:
            doc_text = f"Tipo: Propiedad\nURI: {row.uri}\nEtiqueta: {row.label}\nDescripción: {row.comment}\nDominio (Domain): {row.domain}\nRango (Range): {row.range}"
            documents.append(doc_text)
            count_props += 1

    print(f"  ... {count_props} documentos de Propiedades creados.")

    if len(documents) == 0:
        print("ADVERTENCIA: No se extrajo ningún documento (0 Clases, 0 Propiedades).")
        exit()
        
    print(f"Extracción completada. Se han extraído {len(documents)} documentos en total.")
    print("-" * 30)

    # PASO 4: CONFIGURAR EMBEDDINGS Y CREAR DB
    print("Paso 4: Configurando embeddings y creando la base de datos persistente...")
    
    print("Cargando modelo de embeddings (BAAI/bge-m3)...")
    model_name = "BAAI/bge-m3"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print("Modelo de embeddings cargado.")

    print(f"Creando la base de datos vectorial en '{persist_directory}'...")

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
    print("Ya puedes ejecutar 'rag_basico.py' para hacer consultas.")

except Exception as e:
    print("\n\n--- ERROR CRÍTICO DURANTE LA INDEXACIÓN ---")
    print(f"El script falló. Motivo: {e}")
    print("Detalles del error:")
    traceback.print_exc()