# rag_basico.py (Versión con Arquitectura RAG de 3 Etapas - Selección de Ontologías)
print("Paso 1: Importando librerías...")

import os
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

print("Librerías importadas.")
print("-" * 30)

# PASO 2: CONFIGURACIÓN INICIAL
print("Paso 2: Configurando componentes...")

# 2.1 - Embeddings
print("Configurando embeddings con Sentence Transformers...")
model_name = "BAAI/bge-m3"
embeddings = HuggingFaceEmbeddings(model_name=model_name)
print(f"Embeddings ({model_name}) listos.")

# 2.2 - Vector Store
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
    
    # Verificación
    db_count = vectorstore._collection.count()
    if db_count == 0:
        print(f"\n--- ERROR ---")
        print("La base de datos se ha cargado pero está VACÍA (0 documentos).")
        print("Por favor, ejecuta 'chroma_db.py' primero.")
        exit()
        
    print(f"Base de datos vectorial cargada con éxito. Contiene {db_count} documentos.")

except Exception as e:
    print(f"\n--- ERROR ---")
    print(f"No se pudo cargar la base de datos desde '{persist_directory}'.")
    print(f"Detalle del error: {e}")
    exit()

# 2.3 - LLM
print("Configurando LLM (Llama3 vía Ollama)...")
llm = ChatOllama(model="llama3", temperature=0.0)
print("LLM configurado.")

print("-" * 30)

# PASO 3: DEFINIR PROMPTS PARA LAS 3 ETAPAS

# ETAPA 1: Extracción de Conceptos
extraction_template = """
Eres un experto en análisis semántico. Tu tarea es extraer únicamente las palabras clave de búsqueda de la petición del usuario.

Petición del usuario: **{user_request}**

Instrucciones:
1. Identifica los conceptos principales (Clases) y las relaciones (Propiedades) que el usuario necesita modelar.
2. Extrae SOLO las palabras clave relevantes para buscar en una base de datos de ontologías.
3. Responde con una lista concisa de términos de búsqueda separados por comas.
4. NO proporciones explicaciones, solo las palabras clave.

Ejemplo de respuesta: "Author, Paper, Person, Publication, writes, hasAuthor"

Palabras clave de búsqueda:
"""

extraction_prompt = ChatPromptTemplate.from_template(extraction_template)
extraction_chain = extraction_prompt | llm | StrOutputParser()

# ETAPA 3: Selección y Decisión
selection_template = """
Eres un experto en selección de ontologías. Tu tarea es analizar los resultados de búsqueda y recomendar UNA única ontología.

Petición original del usuario: **{user_request}**

Resultados de búsqueda con sus fuentes:
---
{context_with_sources}
---

Instrucciones:
1. Analiza qué ontología (Fuente) contiene MÁS conceptos relevantes para la petición del usuario.
2. Cuenta el número de conceptos útiles por cada ontología.
3. Evalúa la calidad y completitud de los conceptos encontrados.
4. Recomienda UNA única ontología que mejor cubra los requisitos del usuario.
5. Explica brevemente por qué esa ontología es la mejor opción (menciona qué conceptos clave contiene).

Tu respuesta debe tener este formato:
**ONTOLOGÍA RECOMENDADA:** [nombre_del_archivo]
**RAZÓN:** [explicación breve de 2-3 líneas]
**CONCEPTOS CLAVE ENCONTRADOS:** [lista de 3-5 conceptos principales]

Respuesta:
"""

selection_prompt = ChatPromptTemplate.from_template(selection_template)
selection_chain = selection_prompt | llm | StrOutputParser()

print("Prompts de las 3 etapas configurados.")
print("-" * 30)

# PASO 4: BUCLE PRINCIPAL DE CONSULTAS
print("Paso 4: Sistema de Selección de Ontologías listo.")
print("\n¡Bienvenido al Sistema de Selección de Ontologías!")
print("Describe qué datos necesitas modelar y te recomendaré la mejor ontología.")
print("-" * 60)

while True:
    user_request = input("\n¿Qué necesitas modelar? (o 'salir' para terminar): ")
    
    if user_request.lower() == 'salir':
        break
    
    print("\n" + "=" * 60)
    print("ETAPA 1: EXTRACCIÓN DE CONCEPTOS")
    print("=" * 60)
    
    try:
        # ETAPA 1: Extraer palabras clave con el LLM
        print(" Analizando tu petición para extraer conceptos clave...")
        search_query = extraction_chain.invoke({"user_request": user_request})
        print(f"\n Palabras clave extraídas:\n   {search_query}")
        
    except Exception as e:
        print(f"\n ERROR en la extracción de conceptos: {e}")
        continue
    
    print("\n" + "=" * 60)
    print("ETAPA 2: BÚSQUEDA EN BASE DE DATOS")
    print("=" * 60)
    
    try:
        # ETAPA 2: Buscar en el vectorstore usando las palabras clave
        print(f" Buscando conceptos relacionados con: '{search_query}'...")
        retrieved_docs = vectorstore.max_marginal_relevance_search(search_query, k=15, fetch_k=50)
        
        if not retrieved_docs:
            print("\n No se encontraron resultados relevantes.")
            continue
        
        print(f" Se encontraron {len(retrieved_docs)} documentos relevantes.")
        
        # Construir contexto con fuentes
        context_with_sources = []
        source_count = {}
        
        for i, doc in enumerate(retrieved_docs):
            source = doc.metadata.get('source', 'Fuente desconocida')
            
            # Contar documentos por fuente
            source_count[source] = source_count.get(source, 0) + 1
            
            # Formatear el documento con su fuente
            content_preview = doc.page_content[:200].replace('\n', ' ')
            formatted_doc = f"- [Fuente: {source}] {content_preview}..."
            context_with_sources.append(formatted_doc)
        
        context_str = "\n".join(context_with_sources)
        
        # Mostrar estadísticas de fuentes
        print("\n Distribución de resultados por ontología:")
        for source, count in sorted(source_count.items(), key=lambda x: x[1], reverse=True):
            print(f"   • {source}: {count} documentos")
        
    except Exception as e:
        print(f"\n ERROR en la búsqueda: {e}")
        continue
    
    print("\n" + "=" * 60)
    print("ETAPA 3: SELECCIÓN DE ONTOLOGÍA")
    print("=" * 60)
    
    try:
        # ETAPA 3: Analizar y seleccionar la mejor ontología
        print(" Analizando resultados para seleccionar la mejor ontología...")
        
        recommendation = selection_chain.invoke({
            "user_request": user_request,
            "context_with_sources": context_str
        })
        
        print("\n" + "🎯" * 30)
        print(recommendation)
        print("🎯" * 30)
        
    except Exception as e:
        print(f"\n ERROR en la selección: {e}")
        continue

print("\n¡Hasta luego! ")