print("Iniciando el proceso de indexación (Versión Mejorada)")

import rdflib
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import traceback

print("Paso 1: Librerías importadas.")
print("-" * 30)

persist_directory = "tfg_rag_pruebas/chroma_db"

# Mapeo inicial de formatos
FORMAT_MAP = {
    '.ttl': 'turtle',
    '.n3': 'n3',
    '.owl': 'xml',
    '.rdf': 'xml',
    '.nt': 'nt'
}

def get_safe_value(row, attr_list):
    """Intenta obtener valor de varios atributos posibles de la fila SPARQL"""
    for attr in attr_list:
        if hasattr(row, attr) and getattr(row, attr):
            return str(getattr(row, attr))
    return ""

try:
    print("Paso 2: Procesando ontologías individualmente desde 'dataset'...")
    ontologies_dir = "tfg_rag_pruebas/dataset"
    
    if not os.path.exists(ontologies_dir):
        print(f"ERROR: El directorio 'dataset' no se encuentra en: {os.path.abspath(ontologies_dir)}")
        exit()
    
    all_documents = []
    all_metadatas = []
    
    # Consultas SPARQL ampliadas para capturar SKOS y DC
    query_classes = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX dc: <http://purl.org/dc/elements/1.1/>
        
        SELECT ?uri ?label ?label_skos ?comment ?def_skos ?desc_dc
        WHERE {
            { ?uri a rdfs:Class . } UNION { ?uri a owl:Class . }
            FILTER(!isblank(?uri))
            
            OPTIONAL { ?uri rdfs:label ?label . }
            OPTIONAL { ?uri skos:prefLabel ?label_skos . }
            OPTIONAL { ?uri rdfs:comment ?comment . }
            OPTIONAL { ?uri skos:definition ?def_skos . }
            OPTIONAL { ?uri dc:description ?desc_dc . }
        }
    """
    
    query_properties = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX dc: <http://purl.org/dc/elements/1.1/>

        SELECT ?uri ?label ?label_skos ?comment ?def_skos ?desc_dc ?domain ?range
        WHERE {
            { ?uri a rdf:Property . } UNION { ?uri a owl:ObjectProperty . } UNION { ?uri a owl:DatatypeProperty . }
            FILTER(!isblank(?uri))

            OPTIONAL { ?uri rdfs:label ?label . }
            OPTIONAL { ?uri skos:prefLabel ?label_skos . }
            OPTIONAL { ?uri rdfs:comment ?comment . }
            OPTIONAL { ?uri skos:definition ?def_skos . }
            OPTIONAL { ?uri dc:description ?desc_dc . }
            OPTIONAL { ?uri rdfs:domain ?domain . }
            OPTIONAL { ?uri rdfs:range ?range . }
        }
    """
    
    files_processed = 0
    files_skipped = 0
    
    for filename in os.listdir(ontologies_dir):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in FORMAT_MAP:
            continue
        
        filepath = os.path.join(ontologies_dir, filename)
        print(f"\n--- Procesando: {filename} ---")
        
        g = rdflib.Graph()
        loaded = False
        
        # INTENTO 1: Formato por extensión
        try:
            file_format = FORMAT_MAP[file_ext]
            g.parse(filepath, format=file_format)
            loaded = True
        except Exception as e:
            # INTENTO 2: Si falla XML, probar Turtle 
            if file_format == 'xml':
                try:
                    print(f"  Falló carga XML. Reintentando como Turtle...")
                    g.parse(filepath, format='turtle')
                    loaded = True
                    print(f"  Recuperado exitosamente como Turtle.")
                except:
                    pass # Falló el reintento
            
            if not loaded:
                print(f"  ERROR CRÍTICO al parsear {filename}: {e}")
                files_skipped += 1
                continue

        # Procesamiento si cargó bien
        print(f"  Archivo cargado: {len(g)} tripletas")
        file_documents = []
        
            # CLASES
        try:
            results_classes = g.query(query_classes)
            count_cls = 0
            for row in results_classes:
                # Lógica de "Mejor Etiqueta Disponible"
                label = get_safe_value(row, ['label', 'label_skos'])
                
                # Si no tiene etiqueta, USAR EL URI (Fragmento)
                if not label:
                    label = str(row.uri).split('#')[-1].split('/')[-1]
                
                desc = get_safe_value(row, ['comment', 'def_skos', 'desc_dc'])
                
                doc_text = f"Tipo: Clase\nURI: {row.uri}\nEtiqueta: {label}\nDescripción: {desc}"
                file_documents.append(doc_text)
                count_cls += 1
            print(f"    ... {count_cls} clases extraídas")
        except Exception as e:
             print(f"    Error en query de Clases: {e}")

        # PROPIEDADES 
        try:
            results_props = g.query(query_properties)
            count_prop = 0
            for row in results_props:
                label = get_safe_value(row, ['label', 'label_skos'])
                if not label:
                    label = str(row.uri).split('#')[-1].split('/')[-1]
                    
                desc = get_safe_value(row, ['comment', 'def_skos', 'desc_dc'])
                domain = get_safe_value(row, ['domain'])
                range_ = get_safe_value(row, ['range'])

                doc_text = f"Tipo: Propiedad\nURI: {row.uri}\nEtiqueta: {label}\nDescripción: {desc}\nDominio: {domain}\nRango: {range_}"
                file_documents.append(doc_text)
                count_prop += 1
            print(f"    ... {count_prop} propiedades extraídas")
        except Exception as e:
             print(f"    Error en query de Propiedades: {e}")

        # Guardar
        if file_documents:
            file_metadatas = [{"source": filename} for _ in file_documents]
            all_documents.extend(file_documents)
            all_metadatas.extend(file_metadatas)
            files_processed += 1
            print(f"  {filename}: {len(file_documents)} docs generados")
        else:
            print(f"  ADVERTENCIA: {filename} procesado pero 0 documentos generados.")

    print("\n" + "-" * 30)
    print(f"Resumen Final: {len(all_documents)} documentos totales listos para indexar.")
    
    if len(all_documents) == 0:
        exit()

    # PASO 3: DB
    print("\nPaso 3: Creando DB Vectorial...")
    model_name = "BAAI/bge-m3"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # Limpiar DB anterior (opcional, aquí asumo sobreescritura o append)
    # shutil.rmtree(persist_directory) 
    
    vectorstore = Chroma.from_texts(
        texts=all_documents,
        embedding=embeddings,
        metadatas=all_metadatas,
        persist_directory=persist_directory
    )
    print("Base de datos actualizada.")

except Exception as e:
    traceback.print_exc()