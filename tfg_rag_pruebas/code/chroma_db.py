# (Versión REFACTORIZADA - Procesamiento Individual con Metadatos)
print("Iniciando el proceso de indexación")

import rdflib
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import traceback

print("Paso 1: Librerías importadas.")
print("-" * 30)

# --- Define el directorio de la base de datos primero ---
persist_directory = "tfg_rag_pruebas/chroma_db"

# Mapeo de extensiones a formatos de rdflib
FORMAT_MAP = {
    '.ttl': 'turtle',
    '.n3': 'n3',
    '.owl': 'xml',
    '.rdf': 'xml',
    '.nt': 'nt'
}

try:
    # PASO 2: PROCESAR ONTOLOGÍAS UNA A UNA
    print("Paso 2: Procesando ontologías individualmente desde 'dataset'...")
    ontologies_dir = "tfg_rag_pruebas/dataset"
    
    if not os.path.exists(ontologies_dir):
        print(f"ERROR: El directorio 'dataset' no se encuentra en: {os.path.abspath(ontologies_dir)}")
        exit()
    
    # Listas acumuladoras para todos los documentos y metadatos
    all_documents = []
    all_metadatas = []
    
    # Queries SPARQL (definidas fuera del bucle para reutilizarlas)
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
            
            BIND(COALESCE(?label_rdfs, "") AS ?label)
            BIND(COALESCE(?comment_rdfs, "") AS ?comment)
        }
    """
    
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
    
    files_processed = 0
    files_skipped = 0
    
    # Iterar sobre cada archivo en el directorio
    for filename in os.listdir(ontologies_dir):
        # Verificar si la extensión está soportada
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in FORMAT_MAP:
            continue
        
        filepath = os.path.join(ontologies_dir, filename)
        print(f"\n--- Procesando: {filename} ---")
        
        try:
            # Crear un grafo nuevo para ESTE archivo
            g = rdflib.Graph()
            
            # Cargar el archivo con el formato apropiado
            file_format = FORMAT_MAP[file_ext]
            print(f"  Cargando como formato '{file_format}'...")
            g.parse(filepath, format=file_format)
            
            tripletas_count = len(g)
            print(f"  Archivo cargado: {tripletas_count} tripletas")
            
            if tripletas_count == 0:
                print(f"  ADVERTENCIA: {filename} está vacío. Saltando...")
                files_skipped += 1
                continue
            
            # Ejecutar consultas SPARQL para ESTE grafo
            file_documents = []
            
            # Procesar Clases
            print("  Extrayendo Clases...")
            results_classes = g.query(query_classes)
            classes_count = 0
            for row in results_classes:
                if row.label or row.comment:
                    doc_text = f"Tipo: Clase\nURI: {row.uri}\nEtiqueta: {row.label}\nDescripción: {row.comment}"
                    file_documents.append(doc_text)
                    classes_count += 1
            print(f"    ... {classes_count} clases extraídas")
            
            # Procesar Propiedades
            print("  Extrayendo Propiedades...")
            results_properties = g.query(query_properties)
            props_count = 0
            for row in results_properties:
                if row.label or row.comment:
                    doc_text = f"Tipo: Propiedad\nURI: {row.uri}\nEtiqueta: {row.label}\nDescripción: {row.comment}\nDominio (Domain): {row.domain}\nRango (Range): {row.range}"
                    file_documents.append(doc_text)
                    props_count += 1
            print(f"    ... {props_count} propiedades extraídas")
            
            # Generar metadatos para TODOS los documentos de este archivo
            file_metadatas = [{"source": filename} for _ in file_documents]
            
            # Acumular en las listas globales
            all_documents.extend(file_documents)
            all_metadatas.extend(file_metadatas)
            
            files_processed += 1
            print(f"  ✓ {filename} completado: {len(file_documents)} documentos generados")
            
        except Exception as e:
            print(f"  ✗ ERROR al procesar {filename}: {e}")
            print("  Detalles:")
            traceback.print_exc()
            files_skipped += 1
            continue
    
    print("\n" + "-" * 30)
    print(f"Resumen de carga:")
    print(f"  - Archivos procesados exitosamente: {files_processed}")
    print(f"  - Archivos saltados/con error: {files_skipped}")
    print(f"  - Total de documentos extraídos: {len(all_documents)}")
    
    if len(all_documents) == 0:
        print("\nERROR: No se extrajo ningún documento. Verifica el contenido del directorio 'dataset'.")
        exit()
    
    print("-" * 30)

    # PASO 3: CONFIGURAR EMBEDDINGS Y CREAR DB CON METADATOS
    print("\nPaso 3: Configurando embeddings y creando la base de datos persistente...")
    
    print("Cargando modelo de embeddings (BAAI/bge-m3)...")
    model_name = "BAAI/bge-m3"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print("Modelo de embeddings cargado.")

    print(f"Creando la base de datos vectorial en '{persist_directory}'...")

    # Crear la base de datos CON METADATOS
    vectorstore = Chroma.from_texts(
        texts=all_documents,
        embedding=embeddings,
        metadatas=all_metadatas,  # ← METADATOS AÑADIDOS
        persist_directory=persist_directory
    )
    
    # Forzar el guardado en disco
    # vectorstore.persist()

    print("\n--- ¡ÉXITO! ---")
    print(f"Base de datos vectorial creada y guardada en '{persist_directory}'.")
    
    # Comprobación final
    count = vectorstore._collection.count()
    print(f"Comprobación: La base de datos contiene {count} documentos indexados.")
    print("Ya puedes ejecutar 'rag_basico.py' para hacer consultas.")

except Exception as e:
    print("\n\n--- ERROR CRÍTICO DURANTE LA INDEXACIÓN ---")
    print(f"El script falló. Motivo: {e}")
    print("Detalles del error:")
    traceback.print_exc()