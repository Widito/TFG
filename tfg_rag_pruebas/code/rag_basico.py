import os
import time
import json
import re
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

# CONFIGURACIÓN
PERSIST_DIRECTORY = "tfg_rag_pruebas/chroma_db"
EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL = "llama3"

class OntologyRecommender:
    def __init__(self):
        print("Iniciando sistema RAG")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        if not os.path.exists(PERSIST_DIRECTORY):
            raise FileNotFoundError(f"No se encuentra la BD en {PERSIST_DIRECTORY}")
            
        self.vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY, 
            embedding_function=self.embeddings
        )
        
        self.llm = ChatOllama(model=LLM_MODEL, temperature=0.0)
        self._setup_chains()
        print("Sistema RAG listo.")

    def _setup_chains(self):
        # 1. EXTRACCIÓN 
        extract_tmpl = """
        Analiza la petición del usuario.
        Petición: {user_request}
        
        Tarea: Genera una lista de palabras clave técnicas y conceptos principales para buscar en una base de datos vectorial.
        - Usa terminología técnica (preferiblemente en inglés, ya que es el estándar en ontologías).
        - No limites la cantidad de palabras.
        
        Respuesta: Solo las palabras clave separadas por comas.
        """
        self.extraction_chain = ChatPromptTemplate.from_template(extract_tmpl) | self.llm | StrOutputParser()

        # 2. FILTRADO 
        # CORRECCIÓN: Quitamos el sesgo de "especialización" para evitar borrar ontologías base como BOT.
        filter_tmpl = """
        Eres un filtro de relevancia para un motor de búsqueda de ontologías.
        
        INPUT:
        - Query: "{user_request}"
        - Candidatos: Lista de fragmentos de texto.

        TAREA:
        Selecciona TODOS los archivos que contengan definiciones relevantes para la query.

        CRITERIOS (STRICT):
        1. **Relevancia Temática:** Si el archivo habla del tema de la query (ej. Construcción, Sensores), MANTENLO.
        2. **Cobertura Parcial:** Si el archivo define SOLO UNA PARTE de lo que pide el usuario (ej. define "Zona" pero no "Sensor"), MANTENLO. No busques la perfección, busca utilidad.
        3. **No seas Excluyente:** No descartes una ontología general (Core) solo porque haya una más específica. A menudo se necesitan ambas. Solo descarta lo que sea RUIDO evidente (temas totalmente distintos).

        Candidatos:
        {context_list}

        SALIDA (JSON VÁLIDO):
        {{
            "relevant_sources": ["archivo1.ttl", "archivo2.n3"]
        }}
        """
        self.filter_chain = ChatPromptTemplate.from_template(filter_tmpl) | self.llm | StrOutputParser()

        # 3. DECISIÓN FINAL 
        # CORRECCIÓN: Reintroducimos la prohibición de URIs y conocimiento externo.
        selection_tmpl = """
        Eres un Sistema Recomendador de Ontologías.
        Tu misión es elegir la MEJOR ontología de la lista de candidatos proporcionada.
        
        Petición del Usuario: {user_request}
        
        Contexto (Candidatos Filtrados):
        {filtered_context}
        
        REGLAS DE ORO (A CUMPLIR BAJO PENA DE ERROR):
        1. **GROUNDING TOTAL:** Solo puedes recomendar un archivo que esté en la lista de "Candidatos Filtrados". NO inventes ontologías externas (como CIDOC CRM o Schema.org) si no están en la lista.
        2. **FORMATO DE NOMBRE:** En el campo "ONTOLOGÍA RECOMENDADA", debes escribir EXACTAMENTE el nombre del archivo fuente (ej: 'bot.ttl', 'saref.ttl').
        3. **PROHIBIDO URIs:** NUNCA respondas con una URL o URI (ej: http://w3id.org/bot#...). El usuario necesita el nombre del archivo.
        
        Instrucciones de Decisión:
        - Si la query es sobre topología de edificios (zonas, espacios, plantas), busca ontologías que definan 'Zone', 'Space', 'Storey'.
        - Si hay varias opciones, elige la que tenga definiciones más directas de los términos buscados.
        
        Respuesta:
        **ONTOLOGÍA RECOMENDADA:** [NOMBRE_DEL_ARCHIVO]
        **RAZÓN:** [Justificación breve basada en el contenido del archivo]
        """
        self.selection_chain = ChatPromptTemplate.from_template(selection_tmpl) | self.llm | StrOutputParser()

    def _extract_json_from_text(self, text):
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return json.loads(text)
        except:
            return None

    def run_pipeline(self, user_request, initial_k=40):
        start_time = time.time()
        
        # 1. Extracción
        try: keywords = self.extraction_chain.invoke({"user_request": user_request})
        except: keywords = user_request

        # 2. Retrieval Amplio
        raw_docs = self.vectorstore.max_marginal_relevance_search(keywords, k=initial_k, fetch_k=100)
        
        # Preparar contexto (450 caracteres para ver descripciones)
        doc_summaries = []
        for d in raw_docs:
            src = d.metadata.get('source', 'unknown')
            content = d.page_content[:450].replace('\n', ' ')
            doc_summaries.append(f"- FILE: {src} | CONTENT: {content}...")
        doc_list_str = "\n".join(doc_summaries)
        
        # 3. Re-ranking (Filtrado)
        relevant_files = []
        try:
            raw_filter_output = self.filter_chain.invoke({
                "user_request": user_request,
                "context_list": doc_list_str
            })
            parsed_json = self._extract_json_from_text(raw_filter_output)
            
            if parsed_json and "relevant_sources" in parsed_json:
                relevant_files = parsed_json["relevant_sources"]
            else:
                # Fallback: Top 10 para asegurar variedad si falla el JSON
                relevant_files = [d.metadata.get('source') for d in raw_docs[:10]]
        except Exception as e:
            print(f"Fallback filtro: {e}")
            relevant_files = [d.metadata.get('source') for d in raw_docs[:10]]

        if not isinstance(relevant_files, list): relevant_files = []

        # 4. Reconstrucción Contexto
        final_docs = [d for d in raw_docs if d.metadata.get('source') in relevant_files]
        if not final_docs: final_docs = raw_docs[:5] 

        context_lines = [f"- [Fuente: {d.metadata.get('source')}] {d.page_content[:450]}..." for d in final_docs]
        context_str = "\n".join(context_lines)

        # 5. Generación
        decision_text = self.selection_chain.invoke({
            "user_request": user_request,
            "filtered_context": context_str
        })

        return {
            "query": user_request,
            "keywords": keywords,
            "unique_retrieved_sources": list(set([d.metadata.get('source') for d in final_docs])),
            "llm_response": decision_text,
            "execution_time": time.time() - start_time
        }

if __name__ == "__main__":
    rag = OntologyRecommender()
    while True:
        q = input("\nConsulta ('salir'): ")
        if q == 'salir': break
        res = rag.run_pipeline(q)
        print(f"Filtrados: {res['unique_retrieved_sources']}")
        print(f"Respuesta:\n{res['llm_response']}")