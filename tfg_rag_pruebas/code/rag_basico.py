# PASO 1: IMPORTAR LAS LIBRERÍAS NECESARIAS 
print("Paso 1: Importando librerías...")

import rdflib
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

print("Librerías importadas.")
print("-" * 30)

# PASO 2: CARGAR LA ONTOLOGÍA Y EXTRAER EL TEXTO LEGIBLE
print("Paso 2: Cargando la ontología y extrayendo descripciones...")

# Crear un grafo RDF vacío
g = rdflib.Graph()

# Cargar el fichero de la ontología.
try:
    g.parse("../dataset/mo_2013-07-22.n3", format="n3")
    print(f"Ontología cargada con éxito. Contiene {len(g)} tripletas.")
except Exception as e:
    print(f"Error al cargar la ontología: {e}")
    exit()

# Consulta SPARQL para extraer todas las clases/propiedades y sus comentarios (descripciones)
query_text = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?subject ?comment
    WHERE {
        ?subject rdfs:comment ?comment .
    }
"""
results = g.query(query_text)

# Crear una lista de "documentos". Cada documento será un texto que describe un concepto.
documents = []
for row in results:
    # Filtramos los comentarios para que tengan una longitud mínima, evitando ruido.
    if len(row.comment) > 20:
        doc_text = f"Concepto URI: {row.subject}\nDescripción: {row.comment}"
        documents.append(doc_text)

print(f"Se han extraído {len(documents)} descripciones de la ontología.")
print("-" * 30)

# PASO 3: CONSTRUIR LA CADENA RAG 
print("Paso 3: Construyendo la cadena RAG con LangChain...")

# MODIFICACIÓN: Cambiamos el motor de embeddings a Sentence Transformers para mejor rendimiento.
print("Configurando embeddings con Sentence Transformers (esto puede tardar la primera vez)...")
model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)
print("Embeddings listos.")


# 3.2 - Vector Store: La base de datos que almacena los vectores.
vectorstore = Chroma.from_texts(texts=documents, embedding=embeddings)

# 3.3 - Retriever: El componente que busca en la base de datos vectorial.
# Dado un texto, encontrará los documentos (vectores) más similares.
# k=3 significa que traerá los 3 resultados más relevantes.
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) 
print("Retriever configurado.")

# 3.4 - LLM: El modelo de lenguaje que generará las respuestas.
llm = ChatOllama(model="llama3")

# 3.5 - Prompt Template: La plantilla que le daremos al LLM.
# Le damos instrucciones claras para que use solo el contexto que le proporcionamos.
template = """
Actúa como un asistente experto en la 'Music Ontology'. Responde a la pregunta del usuario basándote ÚNICAMENTE en el siguiente contexto extraído de la ontología. Si la información no está en el contexto, di que no lo sabes.

CONTEXTO:
{context}

PREGUNTA:
{question}

RESPUESTA:
"""
prompt = ChatPromptTemplate.from_template(template)
print("Plantilla de prompt creada.")

# 3.6 - Cadena RAG (RAG Chain): Unimos todas las piezas.
rag_chain = (
    # El diccionario de entrada pasa el contexto (buscado por el retriever) y la pregunta original.
    {"context": retriever, "question": RunnablePassthrough()}
    # Pasamos el diccionario al prompt para que lo formatee.
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

# Bucle interactivo para que puedas hacer varias preguntas.
while True:
    user_question = input("\n ¿Qué quieres saber sobre la Music Ontology?: ")
    if user_question.lower() == 'salir':
        break
    
    print("\n Pensando...")
    # Invocamos la cadena con la pregunta del usuario.
    # El flujo definido en el paso 3.6 se ejecuta automáticamente.
    response = rag_chain.invoke(user_question)

    print("\n Respuesta del LLM:")
    print(response)

print("\n ¡Hasta luego!")