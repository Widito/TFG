import csv
import json
import os
import re
from typing import Dict, List, Optional

from rag_basico import OntologyRecommender as RAGBasico


class EvaluadorRequisitos:
    """Flujo de evaluacion Bottom-Up orientado a requisitos funcionales."""

    def __init__(self, rag: Optional[RAGBasico] = None):
        # Permite inyectar una instancia existente para pruebas o reutilizacion.
        self.rag = rag if rag is not None else RAGBasico()

    def cargar_requisitos(self, ruta_csv: str, max_requirements: Optional[int] = None) -> List[str]:
        """
        Lee un CSV y extrae una lista de requisitos.

        Estrategia de extraccion:
        1) Intenta localizar una columna tipica (requisito/requirement/etc.).
        2) Si no existe, usa la primera columna no vacia por fila.
        3) Aplica limite max_requirements si se indica.
        """
        if not os.path.exists(ruta_csv):
            print(f"[ERROR] No existe el archivo CSV: {ruta_csv}")
            return []

        requisitos: List[str] = []
        candidate_columns = {
            "requisito",
            "requerimiento",
            "requirement",
            "requirements",
            "functional_requirement",
            "functional_requirements",
            "query",
            "consulta",
            "pregunta",
        }

        try:
            with open(ruta_csv, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)

                # Si no hay cabeceras validas, hacemos fallback con csv.reader.
                if not reader.fieldnames:
                    f.seek(0)
                    plain_reader = csv.reader(f)
                    for row in plain_reader:
                        if not row:
                            continue
                        value = str(row[0]).strip()
                        if value:
                            requisitos.append(value)
                    return requisitos[:max_requirements] if max_requirements else requisitos

                # Seleccion de columna de requisitos por nombre.
                normalized = {name.strip().lower(): name for name in reader.fieldnames if name}
                selected_column = None
                for key in candidate_columns:
                    if key in normalized:
                        selected_column = normalized[key]
                        break

                for row in reader:
                    value = ""
                    if selected_column:
                        value = str(row.get(selected_column, "")).strip()
                    else:
                        # Fallback: primera celda no vacia en la fila.
                        for raw in row.values():
                            if raw is not None and str(raw).strip():
                                value = str(raw).strip()
                                break

                    if value:
                        requisitos.append(value)

        except Exception as exc:
            print(f"[ERROR] Fallo leyendo CSV de requisitos: {exc}")
            return []

        if max_requirements is not None and max_requirements > 0:
            return requisitos[:max_requirements]
        return requisitos

    def evaluar_requisito(self, requisito: str, top_k: int = 5) -> List[Dict[str, str]]:
        """
        Recupera candidatos para un requisito usando:
        - Extraccion de keywords con el LLM de RAGBasico.
        - Retrieval hibrido (dense + BM25).
        - Re-ranking con cross-encoder.

        Devuelve una lista de entidades con URI, texto y ontologia fuente.
        """
        if not requisito or not requisito.strip():
            return []

        query = requisito.strip()

        # 1) Extraccion de terminos (si falla, usamos el texto original).
        try:
            keywords = self.rag.extraction_chain.invoke({"user_request": query})
        except Exception:
            keywords = query

        # 2) Retrieval hibrido amplio para dar contexto al reranker.
        broad_k = max(top_k * 5, top_k)
        raw_docs = self.rag._hybrid_retrieve(keywords, k=broad_k)
        if not raw_docs:
            return []

        # 3) Re-ranking con cross-encoder usando pares (requisito, documento).
        pairs = [[query, d.page_content[:500]] for d in raw_docs]
        scores = self.rag.reranker.predict(pairs)

        scored_docs = sorted(zip(raw_docs, scores), key=lambda x: x[1], reverse=True)
        selected = scored_docs[:top_k]

        entidades = []
        for doc, _score in selected:
            texto = doc.page_content
            uri = self._extraer_uri(texto)
            ontologia = doc.metadata.get("source", "unknown") if doc.metadata else "unknown"

            entidades.append(
                {
                    "uri": uri,
                    "texto": texto,
                    "ontologia": ontologia,
                }
            )

        return entidades

    def juez_llm(self, requisito: str, entidades_recuperadas: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Evalua con LLM si cada entidad candidata satisface el requisito funcional.
        Devuelve solo las entidades cuyas URI hayan sido aprobadas por el modelo.
        """
        from langchain_core.prompts import ChatPromptTemplate

        if not requisito or not requisito.strip() or not entidades_recuperadas:
            return []
        
        # 1) Construimos un bloque legible con ID, URI, Ontologia y fragmento textual.
        entidades_formateadas = []
        for idx, entidad in enumerate(entidades_recuperadas):
            uri = str(entidad.get("uri", "")).strip()
            ontologia = str(entidad.get("ontologia", "unknown")).strip()
            texto = str(entidad.get("texto", "")).strip()
            fragmento = re.sub(r"\s+", " ", texto)[:280]
        
            entidades_formateadas.append(
                f"ID: {idx}\nURI: {uri}\nOntologia: {ontologia}\nFragmento: {fragmento}"
            )
        
        bloque_entidades = "\n\n".join(entidades_formateadas)
        
        # 2) Prompt estricto: salida unicamente JSON valido con la clave uris_aprobadas.
        system_prompt = (
            "Eres un Arquitecto de Datos Semanticos experto en ontologias RDF/OWL. "
            "Evalua cada entidad candidata y decide si satisface el requisito funcional del usuario. "
            "Responde unica y exclusivamente con JSON valido. "
            "Prohibido usar markdown, bloques de codigo, saludos o explicaciones adicionales. "
            "Debes devolver exactamente este esquema: "
            '{"uris_aprobadas": ["http://ejemplo/uri1", "http://ejemplo/uri2"]}. '
            "Si ninguna entidad aplica, devuelve: "
            '{"uris_aprobadas": []}.'
        )
        
        human_prompt = (
            "REQUISITO FUNCIONAL:\n{requisito}\n\n"
            "ENTIDADES CANDIDATAS:\n{entidades}\n\n"
            "Devuelve solo el JSON con uris_aprobadas."
        )
        
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", human_prompt),
            ]
        )
        
        prompt_generado = prompt_template.format_messages(
            requisito=requisito.strip(),
            entidades=bloque_entidades,
        )
        
        try:
            # 3) Invocacion real al LLM.
            respuesta = self.rag.llm.invoke(prompt_generado)
            raw_text = respuesta.content if hasattr(respuesta, "content") else str(respuesta)
            raw_text = str(raw_text).strip()
        
            # 4) Limpieza robusta del JSON:
            # - Elimina fences ```json ... ``` si el modelo los incluye indebidamente.
            # - Si hay texto extra antes/despues, extrae el primer bloque {...} parseable.
            cleaned = re.sub(r"^\s*```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.IGNORECASE)
            cleaned = cleaned.strip()
        
            json_candidate = cleaned
            if not (json_candidate.startswith("{") and json_candidate.endswith("}")):
                match = re.search(r"\{[\s\S]*\}", cleaned)
                if not match:
                    return []
                json_candidate = match.group(0).strip()
        
            parsed = json.loads(json_candidate)
            uris_aprobadas = parsed.get("uris_aprobadas", [])
            if not isinstance(uris_aprobadas, list):
                return []
        
            uris_aprobadas_set = {str(uri).strip() for uri in uris_aprobadas if str(uri).strip()}
        
        except Exception:
            # Fallback estricto si falla invocacion o parseo critico.
            return []
        
        # 5) Filtrado final: solo mantenemos entidades cuya URI este aprobada.
        return [
            entidad
            for entidad in entidades_recuperadas
            if str(entidad.get("uri", "")).strip() in uris_aprobadas_set
        ]
    
    def generar_matriz_cobertura(self, tad_requisitos: Dict[str, List[Dict[str, str]]]) -> Dict[str, List[str]]:
        """
        Pivota el TAD requisito->entidades hacia ontologia->requisitos cubiertos.
        """
        cobertura: Dict[str, List[str]] = {}

        for requisito, entidades in tad_requisitos.items():
            for entidad in entidades:
                ontologia = entidad.get("ontologia", "unknown")
                if ontologia not in cobertura:
                    cobertura[ontologia] = []

                if requisito not in cobertura[ontologia]:
                    cobertura[ontologia].append(requisito)

        return cobertura

    def redactar_veredicto_final(self, matriz_cobertura: Dict[str, List[str]]) -> str:
        """
        Genera un veredicto final argumentado recomendando una red de ontologias.

        El LLM recibe la matriz de cobertura en JSON y redacta una recomendacion
        en formato Markdown profesional (no JSON), combinando como maximo 3 ontologias.
        """
        from langchain_core.prompts import ChatPromptTemplate

        matriz_json = json.dumps(matriz_cobertura, indent=2, ensure_ascii=False)

        system_prompt = (
            "Actua como un Consultor Experto en Web Semantica y Ontologias. "
            "Tu tarea es analizar la matriz de cobertura de requisitos por ontologia y proponer "
            "una red de ontologias recomendada. "
            "Debes recomendar un conjunto de maximo 3 ontologias que, juntas, maximicen la cobertura. "
            "Justifica brevemente por que eliges cada ontologia, citando los requisitos que cubre. "
            "La respuesta debe estar en Markdown limpio, claro y profesional. "
            "No devuelvas JSON en esta etapa."
        )

        human_prompt = (
            "MATRIZ DE COBERTURA (JSON):\n{matriz_cobertura_json}\n\n"
            "Redacta el veredicto final con:\n"
            "1) Recomendacion de red (maximo 3 ontologias).\n"
            "2) Justificacion breve por ontologia.\n"
            "3) Observaciones de cobertura y posibles huecos."
        )

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", human_prompt),
            ]
        )

        prompt_generado = prompt_template.format_messages(
            matriz_cobertura_json=matriz_json,
        )

        try:
            respuesta = self.rag.llm.invoke(prompt_generado)
            veredicto = respuesta.content if hasattr(respuesta, "content") else str(respuesta)
            return str(veredicto).strip()
        except Exception as exc:
            return f"No fue posible generar el veredicto final en este momento: {exc}"

    def orquestar_evaluacion(self, ruta_csv: str, max_requirements: int = 5) -> Dict[str, object]:
        """
        Flujo principal Bottom-Up:
        1) Carga requisitos.
        2) Recupera entidades candidatas por requisito.
        3) Filtra entidades con el juez LLM.
        4) Construye TAD requisito->entidades validas.
        5) Genera y muestra matriz de cobertura ontologia->requisitos.
        6) Redacta veredicto final con recomendacion de red de ontologias.
        """
        requisitos = self.cargar_requisitos(ruta_csv, max_requirements=max_requirements)
        if not requisitos:
            print("[INFO] No hay requisitos para evaluar.")
            return {"tad_requisitos": {}, "matriz_cobertura": {}, "veredicto_final": ""}

        tad_requisitos: Dict[str, List[Dict[str, str]]] = {}

        for idx, requisito in enumerate(requisitos, start=1):
            print(f"\n[{idx}/{len(requisitos)}] Evaluando requisito: {requisito[:120]}")

            entidades_recuperadas = self.evaluar_requisito(requisito, top_k=5)
            entidades_validadas = self.juez_llm(requisito, entidades_recuperadas)

            tad_requisitos[requisito] = entidades_validadas

        matriz_cobertura = self.generar_matriz_cobertura(tad_requisitos)

        print("\n=== Matriz de Cobertura (Ontologia -> Requisitos) ===")
        print(json.dumps(matriz_cobertura, indent=2, ensure_ascii=False))

        veredicto_final = self.redactar_veredicto_final(matriz_cobertura)
        print("\n=== Veredicto Final Recomendado ===")
        print(veredicto_final)

        return {
            "tad_requisitos": tad_requisitos,
            "matriz_cobertura": matriz_cobertura,
            "veredicto_final": veredicto_final,
        }

    @staticmethod
    def _extraer_uri(texto_documento: str) -> str:
        """Extrae la URI desde la linea 'URI: ...' del texto almacenado en Chroma."""
        if not texto_documento:
            return ""

        match = re.search(r"^URI:\s*(\S+)", texto_documento, flags=re.MULTILINE)
        if match:
            return match.group(1)
        return ""


if __name__ == "__main__":
    # Ejemplo minimo de ejecucion.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    ruta_default = os.path.join(project_root, "dataset_bot_test.csv")

    evaluador = EvaluadorRequisitos()
    evaluador.orquestar_evaluacion(ruta_default, max_requirements=5)
