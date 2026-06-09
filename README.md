# ontology_rag_evaluator

Una librería de Python modular y portable para la recuperación, recomendación y evaluación funcional de ontologías en sistemas RAG (Retrieval-Augmented Generation). Combina búsqueda vectorial (ChromaDB + `BGE-M3`), búsqueda léxica (`BM25`) y re-ranking con Cross-Encoder (`MiniLM`).

---

##  Requisitos Previos

Antes de utilizar esta librería, asegúrate de cumplir con lo siguiente:

1. **Python:** Versión `>= 3.10` instalada.
2. **Ollama:** El recomendador y evaluador requieren de un servidor Ollama en ejecución local.
   * [Descargar Ollama](https://ollama.com/) e instalar en tu sistema operativo.
   * Descargar el modelo por defecto (`llama3`):
     ```bash
     ollama run llama3
     ```
     *(Asegúrate de que Ollama esté corriendo de fondo en el puerto por defecto `11434`)*.

---

##  Ejecución 100% Offline (Local)

Esta librería está diseñada para ser **completamente autónoma y local**:
* **Primer arranque (Online obligatorio):** La primera vez que ejecutes una consulta, indexación o evaluación, la librería requiere conexión a internet para descargar de Hugging Face los modelos de embeddings (`bge-m3`) y de reordenación (`MiniLM`). Esta descarga es de un solo uso y se realiza automáticamente en segundo plano.
* **Uso posterior (Offline 100%):** Una vez realizada la descarga inicial, los modelos quedan guardados en tu caché local. El sistema se ejecutará de forma estrictamente local y offline, sin enviar consultas de metadatos a la web, optimizando la latencia de respuesta.
* **Ollama:** Las inferencias de lenguaje con Ollama y `llama3` se realizan de manera local en tu máquina y nunca requieren conexión.

---


##  Instalación

Puedes instalar la librería directamente de manera local o a través de un repositorio Git.

### Instalación Local (Entorno de desarrollo)
Navega a la carpeta de la librería y ejecuta:
```bash
pip install -e .
```

### Instalación Directa desde GitHub (Para usuarios)
Puedes instalar la librería en cualquier otro proyecto ejecutando:
```bash
pip install git+https://github.com/Widito/TFG.git#subdirectory=tfg_rag_pruebas
```


---

##  Interfaz de Consola (CLI)

Al instalar la librería, se registra un comando global llamado `ontology-rag` en el sistema. Soporta tres subcomandos principales:

### 1. Indexar Ontologías (`index`)
Escanea directorios en busca de ontologías RDF (`.ttl`, `.n3`, `.owl`, `.rdf`, `.nt`), procesa sus clases y propiedades, y genera el índice en ChromaDB:
```bash
ontology-rag index --src ./dataset ./gov_acad_dataset --db ./chroma_db
```

### 2. Consultar el Recomendador (`query`)
Lanza una petición en lenguaje natural para obtener una recomendación de ontología final:
```bash
ontology-rag query "Necesito clases para medir el consumo de energía en habitaciones" --db ./chroma_db
```

### 3. Evaluar un Dataset de Requisitos (`evaluate`)
Ejecuta el pipeline de evaluación automatizada sobre un dataset de requisitos en CSV, validando candidatos con el Juez LLM y exportando resultados:
```bash
ontology-rag evaluate --csv ./dataset_bot_test.csv --db ./chroma_db --output ./resultado --max-reqs 5
```
*Las trazas de la evaluación se guardarán automáticamente en `./resultado/trazas_ejecucion.json`.*

> [!TIP]
> Puedes añadir el flag global `--debug` antes de cualquier subcomando (ej. `ontology-rag --debug query ...`) para ver logs detallados de la inicialización de modelos, distancias de búsqueda y tiempos de respuesta.

---

##  Uso desde Código Python

También puedes importar la librería en tus propios scripts de Python para integrarla en cualquier otra aplicación.

### Indexación
```python
from ontology_rag import OntologyIndexer

indexer = OntologyIndexer(
    source_directories=["./dataset"],
    persist_directory="./chroma_db",
    embedding_model="BAAI/bge-m3"
)
indexer.build_index()
```

### Consulta Individual (RAG)
```python
from ontology_rag import OntologyRecommender

recommender = OntologyRecommender(
    persist_directory="./chroma_db",
    embedding_model="BAAI/bge-m3",
    llm_model="llama3"
)
response = recommender.run_pipeline("I need room temperature classes")
print(response["llm_response"])
```

### Evaluación en Lote (Benchmark)
```python
from ontology_rag import EvaluadorRequisitos

evaluador = EvaluadorRequisitos(
    persist_directory="./chroma_db",
    llm_model="llama3",
    output_dir="./resultado"
)
evaluador.orquestar_evaluacion(
    ruta_csv="./dataset_bot_test.csv",
    max_requirements=10
)
```

---

##  Inyección de LLMs Personalizados

Si no deseas usar Ollama en local y prefieres utilizar APIs comerciales (como GPT-4 de OpenAI), puedes inyectar cualquier modelo de chat compatible con LangChain:

```python
from langchain_openai import ChatOpenAI
from ontology_rag import OntologyRecommender

# Instanciar el cliente del modelo comercial
gpt_model = ChatOpenAI(model="gpt-4o", api_key="tu-api-key")

# Inyectarlo en el recomendador RAG
recommender = OntologyRecommender(
    persist_directory="./chroma_db",
    llm=gpt_model
)
```
