# rag_basico.py (Versión con Verificación)
print("Paso 1: Importando librerías...")

import os
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

print("Librerías importadas.")
print("-" * 30)

# PASO 3: CONSTRUIR LA CADENA RAG 
print("Paso 3: Construyendo la cadena RAG con LangChain...")

# 3.1 - Embeddings:
print("Configurando embeddings con Sentence Transformers...")
model_name = "BAAI/bge-m3"
embeddings = HuggingFaceEmbeddings(model_name=model_name)
print(f"Embeddings ({model_name}) listos.")

# 3.2 - Vector Store:
persist_directory = "tfg_rag_pruebas/chroma_db"
print(f"Cargando la base de datos vectorial desde '{persist_directory}'...")

if not os.path.exists(persist_directory):
    print(f"\n--- ERROR ---")
    print(f"No se encuentra el directorio de la base de datos: '{persist_directory}'")
    print("¡Debes ejecutar 'chroma_db.py' primero!")
    exit()

try:
    vectorstore = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embeddings
    )
    
    # --- ¡COMPROBACIÓN DE VERIFICACIÓN! ---
    db_count = vectorstore._collection.count()
    if db_count == 0:
        print(f"\n--- ERROR ---")
        print("La base de datos se ha cargado pero está VACÍA (0 documentos).")
        print("Esto significa que 'chroma_db.py' falló.")
        print("Por favor, borra el directorio 'chroma_db' y vuelve a ejecutar el script 1.")
        exit()
        
    print(f"Base de datos vectorial cargada con éxito. Contiene {db_count} documentos.")
    # --- FIN DE LA COMPROBACIÓN ---

except Exception as e:
    print(f"\n--- ERROR ---")
    print(f"No se pudo cargar la base de datos desde '{persist_directory}'.")
    print(f"Detalle del error: {e}")
    exit()


# 3.3 - Retriever
print("Configurando el retriever semántico (Chroma)...")
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
print("Retriever semántico configurado (k=5).")


# 3.4 - LLM:
llm = ChatOllama(model="phi3", temperature=0.0)

# 3.5 - Prompt Template
template = """
Actúa como un experto en modelado semántico y RDF. Tu objetivo es ayudar al usuario a modelar sus datos usando las ontologías de LOV.

La petición del usuario es: **{question}**

Has buscado en la base de datos y este es el ÚNICO CONTEXTO relevante que has encontrado:
---
{context}
---

Por favor, sigue estas reglas ESTRICTAMENTE antes de generar tu respuesta:
1.  Basa tu respuesta SOLAMENTE en las Clases y Propiedades (incluyendo sus prefijos y URIs) que aparecen en el CONTEXTO de arriba.
2.  NO inventes, adivines ni añadas ninguna clase o propiedad que no esté explícitamente listada en el CONTEXTO.
3.  Si las Clases o Propiedades en el CONTEXTO no son suficientes para responder a la petición del usuario, tu ÚNICA respuesta debe ser: "Basándome estrictamente en el contexto proporcionado, no tengo la información necesaria para modelar esa petición." NO añadas nada más.
4.  Si SÍ tienes información, nombra y describe las tripletas (sujeto, predicado, objeto) que el usuario debería crear.

RESPUESTA:
"""
prompt = ChatPromptTemplate.from_template(template)

print("Plantilla de prompt (Asistente de Modelado) creada.")

# 3.6 - Cadena RAG (RAG Chain)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("Cadena RAG construida y lista para usarse.")
print("-" * 30)

# PASO 4: HACER PREGUNTAS AL SISTEMA 
print("Paso 4: ¡Haciendo preguntas! (Escribe 'salir' para terminar)")

while True:
    user_question = input("\n¿Qué petición de datos quieres modelar? (o 'salir'): ")
    if user_question.lower() == 'salir':
        break
    
    print("\n Pensando...")
    response = rag_chain.invoke(user_question)

    print("\n Respuesta del LLM:")
    print(response)

print("\n ¡Hasta luego!")