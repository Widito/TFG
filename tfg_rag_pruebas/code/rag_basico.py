# PASO 1: IMPORTAR LAS LIBRERÍAS NECESARIAS 
print("Paso 1: Importando librerías...")

import rdflib
import os
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
# MEJORA: Se importa el divisor de texto
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("Librerías importadas.")
print("-" * 30)

# PASO 2: CARGAR TODAS LAS ONTOLOGÍAS DEL DIRECTORIO 'dataset' 
print("Paso 2: Cargando todas las ontologías .n3 desde 'dataset'...")

# Crear un grafo RDF vacío
g = rdflib.Graph()

# MODIFICACIÓN: Cargar todos los archivos .n3 de la carpeta 'dataset' 
ontologies_dir = "tfg_rag_pruebas/dataset" # Directorio principal de datasets
files_loaded = 0
files_failed = 0

print(f"Accediendo al directorio: {ontologies_dir}")

try:
    # Iteramos sobre cada archivo en el directorio
    for filename in os.listdir(ontologies_dir):
        # Comprobamos que es un archivo .n3
        if filename.endswith(".n3"):
            filepath = os.path.join(ontologies_dir, filename)
            
            try:
                print(f"  Cargando {filename}...")
                # Cargamos el archivo .n3 en el grafo 'g'
                g.parse(filepath, format="n3")
                files_loaded += 1
            except Exception as e:
                # Si un archivo falla al parsear, informamos y continuamos
                print(f"    -> ERROR: No se pudo parsear {filename}. Motivo: {e}")
                files_failed += 1

    total_tripletas = len(g)
    print("\n--- Resumen de Carga ---")
    print(f"Archivos .n3 procesados: {files_loaded}")
    print(f"Archivos fallidos: {files_failed}")
    
    if total_tripletas == 0:
         print("ADVERTENCIA: No se cargaron tripletas. ¿La ruta del directorio 'dataset' es correcta?")
    
    print(f"Ontologías cargadas con éxito. El grafo total contiene {total_tripletas} tripletas.")

except Exception as e:
    print(f"Error fatal al acceder al directorio {ontologies_dir}: {e}")
    exit()

# Consulta SPARQL para extraer todas las clases/propiedades y sus comentarios (descripciones)
# Se mejora la consulta para incluir etiquetas opcionales.
# MEJORA: Consulta SPARQL mejorada (con UNION) 
# Ahora se buscan descripciones en rdfs:comment Y TAMBIÉN en dc:description
# También se carga dc:title si rdfs:label no existe.
query_text = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dc: <http://purl.org/dc/elements/1.1/>

    SELECT ?subject ?label ?description
    WHERE {
        { ?subject rdfs:comment ?description . }
        UNION
        { ?subject dc:description ?description . }
        
        OPTIONAL { ?subject rdfs:label ?label . }
        OPTIONAL { ?subject dc:title ?label . }
    }
"""
results = g.query(query_text)

# Crear una lista de "documentos". Cada documento será un texto que describe un concepto.
documents = []
for row in results:
    # Se filtran los comentarios para que tengan una longitud mínima, evitando ruido.
    # Se mejora la legibilidad incluyendo la etiqueta si está disponible.
    # MEJORA: Se reduce el filtro de longitud
    # Se baja de 20 a 5 para incluir descripciones cortas pero válidas.
    if len(row.description) > 5:
        label_text = str(row.label) if row.label else ""
        doc_text = f"Concepto URI: {row.subject}\nEtiqueta: {label_text}\nDescripción: {row.description}"
        documents.append(doc_text)

print(f"Se han extraído {len(documents)} descripciones de la ontología.")
print("-" * 30)

# PASO 3: CONSTRUIR LA CADENA RAG 
print("Paso 3: Construyendo la cadena RAG con LangChain...")

# 3.1 - Embeddings: Convertir texto a vectores.
print("Configurando embeddings con Sentence Transformers...")
model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)
print("Embeddings listos.")

# MEJORA: Se añade el Divisor de Texto (Text Splitter)
print("Dividiendo los documentos en fragmentos (chunks)...")
# MEJORA: Se optimiza el tamaño de los fragmentos ---
# Se crean chunks más pequeños (500) para que sean más específicos.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Tamaño de cada fragmento (más pequeño)
    chunk_overlap=100, # Solapamiento (un poco menos)
    length_function=len
)
document_splits = text_splitter.create_documents(documents)
print(f"Se han creado {len(document_splits)} fragmentos (chunks) para la base de datos.")


# 3.2 - Vector Store: La base de datos que almacena los vectores.
print("Creando la base de datos vectorial con Chroma...")
# MEJORA: Se usa .from_documents() en lugar de .from_texts()
vectorstore = Chroma.from_documents(documents=document_splits, embedding=embeddings)

# 3.3 - Retriever: El componente que busca en la base de datos vectorial.
# Dado un texto, encontrará los documentos (vectores) más similares.
# MEJORA: Se aumenta K a 10 para obtener más contexto.
retriever = vectorstore.as_retriever(search_kwargs={"k": 10}) 
print("Retriever configurado.")

# 3.4 - LLM: El modelo de lenguaje que generará las respuestas.
# MODIFICACION: Se utiliza Ollama con el modelo deepseek-r1:8b.
llm = ChatOllama(model="deepseek-r1:8b")

# 3.5 - Prompt Template
template = """
Actúa como un experto en modelado semántico y RDF. Tu objetivo es ayudar al usuario a modelar sus datos usando las ontologías de LOV.

La petición del usuario es: **{question}**

He buscado en la base de datos de Clases y Propiedades y este es el CONTEXTO relevante que he encontrado:
---
{context}
---

Por favor, basándote ÚNICAMENTE en este contexto:
1.  Identifica las Clases y Propiedades más adecuadas (incluyendo su prefijo).
2.  **Nombra y describe** las tripletas (sujeto, predicado, objeto) que el usuario debería crear para modelar su petición.
3.  Si el contexto no es suficiente para responder, simplemente di que no tienes la información necesaria.

RESPUESTA:
"""
prompt = ChatPromptTemplate.from_template(template)
print("Plantilla de prompt (Asistente de Modelado) creada.")

# 3.6 - Cadena RAG (RAG Chain)
rag_chain = (
    # El diccionario de entrada pasa el contexto (buscado por el retriever) y la pregunta original.
    {"context": retriever, "question": RunnablePassthrough()}
    # Se pasa el diccionario al prompt para que lo formatee.
    | prompt
    # El prompt formateado se pasa al LLM.
    | llm
    # La salida del LLM se convierte a un string simple.
    | StrOutputParser()
)

print("Cadena RAG construida y lista para usarse.")
print("-" * 30)

# PASO 4: HACER PREGUNTAS AL SISTEMA 
print("Paso 4: ¡Haciendo preguntas! (Escribe 'salir' para terminar)")

# Bucle interactivo para poder hacer varias preguntas.
while True:
    user_question = input("\n¿Qué petición de datos quieres modelar? (o 'salir'): ")
    if user_question.lower() == 'salir':
        break
    
    print("\n Pensando...")
    # Se invoca la cadena con la pregunta del usuario.
    # El flujo definido en el paso 3.6 se ejecuta automáticamente.
    response = rag_chain.invoke(user_question)

    print("\n Respuesta del LLM:")
    print(response)

print("\n ¡Hasta luego!")