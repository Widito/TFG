# rag_basico.py (Versión con Verificación)
print("Paso 1: Importando librerías...")

import os
from langchain_chroma import Chroma
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
print("Configurando el retriever semántico (MMR - Chroma)...")
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 10, 'fetch_k': 50} # Pide 50 docs, re-ordena y devuelve los 10 mejores
)
print("Retriever MMR configurado (k=10, fetch_k=50).")


# 3.4 - LLM:
llm = ChatOllama(model="llama3", temperature=0.0)

# 3.5 - Prompt Template
template = """
Actúa como un experto en modelado semántico y RDF. Tu objetivo es ayudar al usuario a modelar sus datos usando las ontologías de LOV.

La petición del usuario es: **{question}**

Has buscado en la base de datos y este es el CONTEXTO relevante que has encontrado:
---
{context}
---

Por favor, sigue estas reglas ESTRICTAMENTE para generar tu respuesta:
1.  Basa tu respuesta **únicamente** en las Clases (ej. "Tipo: Clase") y Propiedades (ej. "Tipo: Propiedad") que aparecen en el CONTEXTO.

2.  **REGLA DE ORO DE RDF (CÓMO USAR PROPIEDADES):**
    * El 'predicado' en una tripleta (sujeto, predicado, objeto) **DEBE** ser una Propiedad (del contexto, ej. "Tipo: Propiedad").
    * **¡MUY IMPORTANTE!** Para modelar una *relación* (como "un Test tiene Preguntas"), busca una Propiedad en el contexto. El `Dominio (Domain)` de esa propiedad te dice qué Clase va en el 'sujeto', y el `Rango (Range)` te dice qué Clase va en el 'objeto'.
    * **NUNCA** uses una Clase (ej. "Tipo: Clase") como si fuera un 'predicado'.
    * **NUNCA** uses `rdfs:subClassOf` para conectar una Clase con una Propiedad.

3.  **REGLA DE PRIORIDAD:** Al analizar el contexto, **prioriza el uso** de Clases y Propiedades que coincidan semántica y textualmente con las palabras clave de la petición del usuario.

4.  Si las Clases y Propiedades relevantes están incompletas, **propón el modelo con lo que tienes** y menciona qué parte de la información no se encontró.

5.  Si, después de aplicar la Regla 3, determinas que ningún documento del contexto es relevante para la petición del usuario, tu ÚNICA respuesta debe ser: "Basándome en el contexto proporcionado, no tengo la información necesaria para modelar esa petición."

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
    
    print("\n Buscando contexto...")
    
    # --- PASO DE DEBUG ---
    try:
        retrieved_docs = retriever.invoke(user_question)
        
        print("--- INICIO: Contexto Recuperado (DEBUG) ---")
        if not retrieved_docs:
            print("ERROR DE DEBUG: El retriever NO ha devuelto NADA.")
        else:
            print(f"El retriever ha encontrado {len(retrieved_docs)} documentos:")
            for i, doc in enumerate(retrieved_docs):
                print(f"\n--- Documento {i+1} (de {len(retrieved_docs)}) ---")
                # Imprime solo los primeros 400 caracteres
                print(doc.page_content[:400] + "...") 
        print("--- FIN: Contexto Recuperado (DEBUG) ---")
    
    except Exception as e:
        print(f"ERROR DE DEBUG al invocar el retriever: {e}")
    # --- FIN DE DEBUG ---


    print("\n Pensando... (Enviando al LLM)")
    # Ahora, invocamos la cadena completa
    response = rag_chain.invoke(user_question)

    print("\n Respuesta del LLM:")
    print(response)

print("\n ¡Hasta luego!")