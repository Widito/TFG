import rdflib
import os
import time

# Configuración
ontologies_dir = "tfg_rag_pruebas/dataset"
FORMAT_MAP = {
    '.ttl': 'turtle',
    '.n3': 'n3',
    '.owl': 'xml',
    '.rdf': 'xml',
    '.nt': 'nt'
}

# Queries SPARQL (Las mismas que usaremos en producción)
QUERY_CLASSES = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT ?uri WHERE {
        { ?uri a rdfs:Class . } UNION { ?uri a owl:Class . }
        FILTER(!isblank(?uri))
    }
"""

QUERY_PROPS = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    SELECT ?uri WHERE {
        { ?uri a rdf:Property . } UNION { ?uri a owl:ObjectProperty . } UNION { ?uri a owl:DatatypeProperty . }
        FILTER(!isblank(?uri))
    }
"""

def debug_ontology(filename):
    filepath = os.path.join(ontologies_dir, filename)
    file_ext = os.path.splitext(filename)[1].lower()
    
    if file_ext not in FORMAT_MAP:
        return
    
    print(f"\n🔎 Analizando: {filename}")
    g = rdflib.Graph()
    start_time = time.time()
    
    try:
        # Intento 1: Formato por extensión
        fmt = FORMAT_MAP[file_ext]
        try:
            g.parse(filepath, format=fmt)
        except Exception as e_primary:
            # Intento 2: Fallback (especialmente para .owl que a veces es Turtle)
            if fmt == 'xml':
                print(f"   ⚠ Falló carga XML ({e_primary}). Probando Turtle...")
                g.parse(filepath, format='turtle')
            else:
                raise e_primary

        # Si llegamos aquí, ha cargado
        triplets = len(g)
        classes = len(list(g.query(QUERY_CLASSES)))
        props = len(list(g.query(QUERY_PROPS)))
        
        duration = time.time() - start_time
        
        # ANÁLISIS DE SALUD
        status = "✅ OK"
        if triplets == 0: status = "❌ VACÍO"
        elif classes == 0 and props == 0: status = "⚠️ SIN DATOS ESTRUCTURALES (¿Faltan prefijos?)"
        
        print(f"   {status} | Tripletas: {triplets} | Clases: {classes} | Props: {props} | Tiempo: {duration:.2f}s")

    except Exception as e:
        print(f"   ❌ ERROR CRÍTICO: {e}")

# Ejecución
print(f"--- INICIANDO DEBUG RÁPIDO en '{ontologies_dir}' ---")
if not os.path.exists(ontologies_dir):
    print("Error: Directorio no encontrado")
    exit()

files = sorted(os.listdir(ontologies_dir))
for f in files:
    debug_ontology(f)

print("\n--- FIN DEL DEBUG ---")