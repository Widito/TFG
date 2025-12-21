import rdflib
import os
import shutil  # Agregado para limpieza opcional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import traceback

print("Iniciando el proceso de indexación (Multi-Folder)")

# --- CONFIGURACIÓN ---
persist_directory = "tfg_rag_pruebas/chroma_db"

# LISTA DE CARPETAS A PROCESAR
# 1. Dataset Objetivo (Tus ontologías buenas)
# 2. Ruido 1 (Gobierno/Academia)
# 3. Ruido 2 (Industria - Hard Negatives)
folders_to_process = [
    "tfg_rag_pruebas/dataset",
    "tfg_rag_pruebas/gov_acad_dataset",
    "tfg_rag_pruebas/dataset_noise_industry"
]

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
    print("Paso 1: Preparando entorno...")
    
    # Listas globales para acumular todo antes de vectorizar
    all_documents = []
    all_metadatas = []
    
    # Consultas SPARQL (INTACTAS)
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
    
    # --- BUCLE PRINCIPAL (ITERANDO POR CARPETAS) ---
    for ontologies_dir in folders_to_process:
        if not os.path.exists(ontologies_dir):
            print(f" ADVERTENCIA: La carpeta '{ontologies_dir}' no existe. Saltando...")
            continue
            
        print(f"\n Procesando carpeta: {ontologies_dir}")
        
        files_processed = 0
        
        for filename in os.listdir(ontologies_dir):
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in FORMAT_MAP:
                continue
            
            filepath = os.path.join(ontologies_dir, filename)
            # print(f"   Analizando: {filename}") 
            
            g = rdflib.Graph()
            loaded = False
            
            # INTENTO 1: Formato por extensión (Lógica original)
            try:
                file_format = FORMAT_MAP[file_ext]
                g.parse(filepath, format=file_format)
                loaded = True
            except Exception as e:
                # INTENTO 2: Fallback a Turtle (Lógica original)
                if file_format == 'xml':
                    try:
                        g.parse(filepath, format='turtle')
                        loaded = True
                    except:
                        pass 
                
                if not loaded:
                    print(f"   ERROR parseando {filename}: {e}")
                    continue

            # Procesamiento (Lógica original)
            file_documents = []
            
            # CLASES
            try:
                results_classes = g.query(query_classes)
                for row in results_classes:
                    label = get_safe_value(row, ['label', 'label_skos'])
                    if not label: label = str(row.uri).split('#')[-1].split('/')[-1]
                    desc = get_safe_value(row, ['comment', 'def_skos', 'desc_dc'])
                    
                    doc_text = f"Tipo: Clase\nURI: {row.uri}\nEtiqueta: {label}\nDescripción: {desc}"
                    file_documents.append(doc_text)
            except Exception: pass

            # PROPIEDADES 
            try:
                results_props = g.query(query_properties)
                for row in results_props:
                    label = get_safe_value(row, ['label', 'label_skos'])
                    if not label: label = str(row.uri).split('#')[-1].split('/')[-1]
                    desc = get_safe_value(row, ['comment', 'def_skos', 'desc_dc'])
                    domain = get_safe_value(row, ['domain'])
                    range_ = get_safe_value(row, ['range'])

                    doc_text = f"Tipo: Propiedad\nURI: {row.uri}\nEtiqueta: {label}\nDescripción: {desc}\nDominio: {domain}\nRango: {range_}"
                    file_documents.append(doc_text)
            except Exception: pass

            # Guardar en acumulador global
            if file_documents:
                # AÑADIDO: 'origin_folder' en metadatos para trazabilidad
                file_metadatas = [{"source": filename, "origin_folder": ontologies_dir} for _ in file_documents]
                all_documents.extend(file_documents)
                all_metadatas.extend(file_metadatas)
                files_processed += 1
        
        print(f"   ✅ {files_processed} archivos procesados en esta carpeta.")

    print("\n" + "-" * 30)
    print(f"Resumen Final: {len(all_documents)} documentos totales de TODAS las carpetas.")
    
    if len(all_documents) == 0:
        print("Error: No se extrajeron documentos.")
        exit()

    # PASO 3: DB (Solo una vez al final)
    print("\nPaso 3: Regenerando DB Vectorial Completa...")
    model_name = "BAAI/bge-m3"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # Limpieza recomendada para Stress Test (evitar duplicados de pruebas anteriores)
    if os.path.exists(persist_directory):
        try:
            shutil.rmtree(persist_directory)
            print("   Base de datos anterior eliminada para inyección limpia.")
        except: pass
    
    # Creación en batch para evitar Memory Errors con tantas ontologías
    batch_size = 5000
    for i in range(0, len(all_documents), batch_size):
        print(f"   Insertando lote {i} a {i+batch_size}...")
        Chroma.from_texts(
            texts=all_documents[i:i+batch_size],
            embedding=embeddings,
            metadatas=all_metadatas[i:i+batch_size],
            persist_directory=persist_directory
        )
        
    print("Base de datos actualizada con éxito.")

except Exception as e:
    traceback.print_exc()