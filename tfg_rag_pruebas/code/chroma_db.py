import rdflib
import os
import shutil
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import traceback

print("Iniciando Indexación Inteligente (Structural RAG)...")

# CONFIGURACIÓN 
# Usamos rutas absolutas para evitar problemas
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
persist_directory = os.path.join(project_root, "chroma_db")

folders_to_process = [
    os.path.join(project_root, "dataset"),
    os.path.join(project_root, "gov_acad_dataset"),
    os.path.join(project_root, "dataset_noise_industry")
]

FORMAT_MAP = {'.ttl': 'turtle', '.n3': 'n3', '.owl': 'xml', '.rdf': 'xml', '.nt': 'nt'}

def analyze_ontology_structure(graph):
    """Analiza la estructura del grafo para determinar si es Core o Extensión"""
    # Contar importaciones explícitas
    query_imports = "SELECT (COUNT(?o) AS ?count) WHERE { ?s <http://www.w3.org/2002/07/owl#imports> ?o }"
    try:
        res = list(graph.query(query_imports))
        import_count = int(res[0][0])
    except:
        import_count = 0
        
    # Heurística simple: Si importa otras ontologías, tiende a ser una extensión/aplicación
    # Si no importa nada (o solo vocabularios básicos no detectados aquí), tiende a ser Core.
    ontology_type = "EXTENSION" if import_count > 0 else "CORE"
    return ontology_type, import_count

def get_safe_value(row, attr_list):
    for attr in attr_list:
        if hasattr(row, attr) and getattr(row, attr):
            return str(getattr(row, attr))
    return ""

try:
    print("Paso 1: Extracción y Análisis Estructural...")
    all_documents = []
    all_metadatas = []
    
    # Queries SPARQL (Mantenemos las tuyas, funcionan bien)
    query_classes = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX dc: <http://purl.org/dc/elements/1.1/>
        SELECT ?uri ?label ?comment ?def_skos WHERE {
            { ?uri a rdfs:Class . } UNION { ?uri a owl:Class . }
            FILTER(!isblank(?uri))
            OPTIONAL { ?uri rdfs:label ?label . }
            OPTIONAL { ?uri rdfs:comment ?comment . }
            OPTIONAL { ?uri skos:definition ?def_skos . }
        }
    """
    
    query_properties = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        SELECT ?uri ?label ?comment ?def_skos ?domain ?range WHERE {
            { ?uri a rdf:Property . } UNION { ?uri a owl:ObjectProperty . } UNION { ?uri a owl:DatatypeProperty . }
            FILTER(!isblank(?uri))
            OPTIONAL { ?uri rdfs:label ?label . }
            OPTIONAL { ?uri rdfs:comment ?comment . }
            OPTIONAL { ?uri skos:definition ?def_skos . }
            OPTIONAL { ?uri rdfs:domain ?domain . }
            OPTIONAL { ?uri rdfs:range ?range . }
        }
    """
    
    for ontologies_dir in folders_to_process:
        if not os.path.exists(ontologies_dir):
            print(f"Saltando carpeta no encontrada: {ontologies_dir}")
            continue
            
        print(f"\n Procesando: {os.path.basename(ontologies_dir)}")
        
        for filename in os.listdir(ontologies_dir):
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in FORMAT_MAP: continue
            
            filepath = os.path.join(ontologies_dir, filename)
            g = rdflib.Graph()
            loaded = False
            
            # Carga robusta
            try:
                g.parse(filepath, format=FORMAT_MAP[file_ext])
                loaded = True
            except:
                if FORMAT_MAP[file_ext] == 'xml': # Fallback XML->Turtle
                    try: g.parse(filepath, format='turtle'); loaded = True
                    except: pass
            
            if not loaded: continue

            # --- ANÁLISIS ESTRUCTURAL (NUEVO) ---
            ont_type, n_imports = analyze_ontology_structure(g)
            # ------------------------------------

            file_docs = []
            
            # Procesar Clases
            for row in g.query(query_classes):
                label = get_safe_value(row, ['label']) or str(row.uri).split('#')[-1]
                desc = get_safe_value(row, ['comment', 'def_skos'])
                # Enriquecemos el texto para BM25
                doc_text = f"Concept: Class\nOntology: {filename} ({ont_type})\nURI: {row.uri}\nLabel: {label}\nDefinition: {desc}"
                file_docs.append(doc_text)

            # Procesar Propiedades
            for row in g.query(query_properties):
                label = get_safe_value(row, ['label']) or str(row.uri).split('#')[-1]
                desc = get_safe_value(row, ['comment', 'def_skos'])
                doc_text = f"Concept: Property\nOntology: {filename} ({ont_type})\nURI: {row.uri}\nLabel: {label}\nDefinition: {desc}"
                file_docs.append(doc_text)

            if file_docs:
                # Guardamos la metadata estructural
                metas = [{
                    "source": filename, 
                    "ontology_type": ont_type, 
                    "imports_count": n_imports,
                    "origin_folder": os.path.basename(ontologies_dir)
                } for _ in file_docs]
                
                all_documents.extend(file_docs)
                all_metadatas.extend(metas)
                # print(f"   -> {filename}: Detectado como {ont_type} ({n_imports} imports)")

    # REGENERACIÓN DB
    print(f"\nRegenerando DB con {len(all_documents)} fragmentos...")
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    
    batch_size = 5000
    for i in range(0, len(all_documents), batch_size):
        print(f"   Lote {i}-{i+batch_size}...")
        Chroma.from_texts(
            texts=all_documents[i:i+batch_size],
            embedding=embeddings,
            metadatas=all_metadatas[i:i+batch_size],
            persist_directory=persist_directory
        )
        
    print("✅ Ingesta Completada con Metadatos Estructurales.")

except Exception as e:
    traceback.print_exc()