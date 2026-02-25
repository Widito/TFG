import rdflib
import os
import time
import sys

# --- CLASE PARA REDIRIGIR SALIDA (LOGGING) ---
class DualLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Asegura que se guarde en tiempo real

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirigir stdout al archivo y pantalla
current_dir = os.path.dirname(os.path.abspath(__file__))
log_filename = os.path.join(current_dir, "log_debug_ontologias.txt")
sys.stdout = DualLogger(log_filename)
# ---------------------------------------------


# Configuración de carpetas (Rutas dinámicas)
project_root = os.path.dirname(current_dir)

folders_to_process = [
    os.path.join(project_root, "dataset"),
    os.path.join(project_root, "gov_acad_dataset"),
    os.path.join(project_root, "dataset_noise_industry")
]

FORMAT_MAP = {
    '.ttl': 'turtle',
    '.n3': 'n3',
    '.owl': 'xml',
    '.rdf': 'xml',
    '.nt': 'nt'
}

# Queries SPARQL
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

def debug_ontology(filename, current_folder_path):
    filepath = os.path.join(current_folder_path, filename)
    file_ext = os.path.splitext(filename)[1].lower()
    
    if file_ext not in FORMAT_MAP:
        return None # No es un formato soportado
    
    print(f"\n Analizando: {filename}")
    g = rdflib.Graph()
    start_time = time.time()
    
    try:
        # Intento 1: Formato por extensión
        fmt = FORMAT_MAP[file_ext]
        try:
            g.parse(filepath, format=fmt)
        except Exception as e_primary:
            # Intento 2: Fallback
            if fmt == 'xml':
                print(f" Falló carga XML ({e_primary}). Probando Turtle...")
                g.parse(filepath, format='turtle')
            else:
                raise e_primary

        # Métricas
        triplets = len(g)
        classes = len(list(g.query(QUERY_CLASSES)))
        props = len(list(g.query(QUERY_PROPS)))
        duration = time.time() - start_time
        
        # ANÁLISIS DE SALUD
        status = "OK"
        if triplets == 0: 
            status = "VACÍO"
        elif classes == 0 and props == 0: 
            status = "SIN DATOS ESTRUCTURALES"
        
        print(f"   {status} | Tripletas: {triplets} | Clases: {classes} | Props: {props} | Tiempo: {duration:.2f}s")
        return status

    except Exception as e:
        print(f"   ERROR CRÍTICO: {e}")
        return "ERROR CRÍTICO"


# ==========================================
# EJECUCIÓN Y RECOPILACIÓN DE ESTADÍSTICAS
# ==========================================
print("--- INICIANDO DEBUG GLOBAL DE ONTOLOGÍAS ---")

# Diccionario para agrupar los resultados
estadisticas = {
    "OK": 0,
    "VACÍO": 0,
    "SIN DATOS ESTRUCTURALES": 0,
    "ERROR CRÍTICO": 0
}
total_archivos = 0

for folder in folders_to_process:
    folder_name = os.path.basename(folder)
    print(f"\n{'='*50}")
    print(f" ESCANEANDO CARPETA: {folder_name}")
    print(f"{'='*50}")
    
    if not os.path.exists(folder):
        print(f" Aviso: Directorio no encontrado ({folder})")
        continue

    files = sorted(os.listdir(folder))
    if not files:
        print("Carpeta vacía.")
        continue
        
    for f in files:
        resultado = debug_ontology(f, folder)
        
        if resultado: # Ignoramos los archivos None (formatos no soportados)
            total_archivos += 1
            if resultado in estadisticas:
                estadisticas[resultado] += 1
            else:
                estadisticas[resultado] = 1

# --- NUEVO: MOSTRAR RESUMEN FINAL ---
print("\n" + "="*50)
print(" RESUMEN FINAL DE CLASIFICACIÓN")
print("="*50)
print(f"Total de ontologías analizadas: {total_archivos}\n")

# Mostrar ordenado y con porcentajes
for estado, cantidad in estadisticas.items():
    if total_archivos > 0:
        porcentaje = (cantidad / total_archivos) * 100
        print(f"  {estado.ljust(25)}: {cantidad} archivos ({porcentaje:.1f}%)")
    else:
        print(f"  {estado.ljust(25)}: {cantidad} archivos")

print("\n--- FIN DEL DEBUG ---")