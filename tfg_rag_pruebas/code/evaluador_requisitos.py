import csv
import json
import os
import re
from typing import Dict, List, Optional, Any

from rag_basico import OntologyRecommender as RAGBasico


class EvaluadorRequisitos:
    """Flujo de evaluacion Bottom-Up orientado a requisitos funcionales."""

    def __init__(self, rag: Optional[RAGBasico] = None):
        # Permite inyectar una instancia existente para pruebas o reutilizacion.
        self.rag = rag if rag is not None else RAGBasico()

    @staticmethod
    def _normalizar_requisito(texto: str) -> str:
        """
        Normaliza un requisito:
        - elimina prefijos numericos (ej: "1;", "2. ", "3) ")
        - limpia espacios multiples
        - recorta espacios extremos
        """
        if texto is None:
            return ""

        requisito = str(texto).strip()
        requisito = re.sub(r"^\s*\d+\s*[\.;:\-\)]\s*", "", requisito)
        requisito = re.sub(r"\s+", " ", requisito).strip()
        return requisito

    @staticmethod
    def _extraer_json_objeto(texto_llm: str) -> Dict[str, Any]:
        """
        Extrae un objeto JSON desde texto libre, usando el primer '{' y el ultimo '}'.
        """
        if not texto_llm:
            raise ValueError("Respuesta vacia del LLM")

        cleaned = re.sub(r"^\s*```(?:json)?\s*", "", str(texto_llm).strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.IGNORECASE).strip()

        first_curly = cleaned.find("{")
        last_curly = cleaned.rfind("}")
        if first_curly == -1 or last_curly == -1 or last_curly <= first_curly:
            raise ValueError("No se encontro bloque JSON delimitado por llaves")

        json_candidate = cleaned[first_curly:last_curly + 1].strip()
        return json.loads(json_candidate)

    def cargar_requisitos(
        self,
        ruta_csv: str,
        max_requirements: Optional[int] = None,
        start_row: Optional[int] = None,
        end_row: Optional[int] = None,
        selected_rows: Optional[List[int]] = None,
        selected_ids: Optional[List[int]] = None,
    ) -> List[str]:
        """
        Lee un CSV y extrae una lista de requisitos.

        Estrategia de extraccion:
        1) Intenta localizar una columna tipica (requisito/requirement/etc.).
        2) Si no existe, usa la primera columna no vacia por fila.
        3) Permite filtrar por rango de filas, filas concretas o IDs del CSV.
        4) Aplica limite max_requirements si se indica.
        """
        if not os.path.exists(ruta_csv):
            print(f"[ERROR] No existe el archivo CSV: {ruta_csv}")
            return []

        requisitos: List[str] = []
        candidate_columns = [
            "query_natural",
            "requisito",
            "requerimiento",
            "requirement",
            "requirements",
            "functional_requirement",
            "functional_requirements",
            "query",
            "consulta",
            "pregunta",
        ]

        try:
            with open(ruta_csv, "r", encoding="utf-8-sig", newline="") as f:
                sample = f.read(1024)
                f.seek(0)

                try:
                    dialect = csv.Sniffer().sniff(f.read(1024))
                    f.seek(0)
                except csv.Error:
                    f.seek(0)
                    dialect = csv.excel
                    dialect.delimiter = ";" if ";" in sample else ","

                reader = csv.DictReader(f, dialect=dialect)

                # Si no hay cabeceras validas, hacemos fallback con csv.reader.
                if not reader.fieldnames:
                    f.seek(0)
                    plain_reader = csv.reader(f, dialect=dialect)
                    for row in plain_reader:
                        if not row:
                            continue
                        value = self._normalizar_requisito(str(row[0]).strip())
                        if len(value) >= 10:
                            requisitos.append(value)
                    return requisitos[:max_requirements] if max_requirements else requisitos

                # Seleccion de columna de requisitos por nombre.
                normalized = {name.strip().lower(): name for name in reader.fieldnames if name}
                selected_column = None
                for key in candidate_columns:
                    if key in normalized:
                        selected_column = normalized[key]
                        break

                excluded_fallback_cols = {"id", "expected_ontology", "difficulty"}

                for row_index, row in enumerate(reader, start=1):
                    csv_id = None
                    if "id" in row and str(row["id"]).strip().isdigit():
                        csv_id = int(str(row["id"]).strip())

                    if start_row is not None and row_index < start_row:
                        continue
                    if end_row is not None and row_index > end_row:
                        continue
                    if selected_rows is not None and row_index not in selected_rows:
                        continue
                    if selected_ids is not None and csv_id is not None and csv_id not in selected_ids:
                        continue

                    value = ""
                    if selected_column:
                        value = str(row.get(selected_column, "")).strip()
                    else:
                        # Fallback robusto: evita columnas tecnicas no funcionales.
                        for col_name, raw in row.items():
                            col_norm = str(col_name).strip().lower() if col_name else ""
                            if col_norm in excluded_fallback_cols:
                                continue
                            if raw is not None and str(raw).strip():
                                value = str(raw).strip()
                                break

                    value = self._normalizar_requisito(value)
                    if len(value) >= 10:
                        requisitos.append(value)

        except Exception as exc:
            print(f"[ERROR] Fallo leyendo CSV de requisitos: {exc}")
            return []

        if max_requirements is not None and max_requirements > 0:
            return requisitos[:max_requirements]
        return requisitos

    def evaluar_requisito(self, requisito: str, top_k: int = 5, return_trace: bool = False):
        """
        Recupera candidatos para un requisito usando:
        - Extraccion de keywords con el LLM de RAGBasico.
        - Retrieval hibrido (dense + BM25).
        - Re-ranking con cross-encoder.

        Devuelve una lista de entidades con URI, texto y ontologia fuente.
        """
        if not requisito or not requisito.strip():
            return {"keywords": "", "entidades_recuperadas": []} if return_trace else []

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
            empty = {"keywords": str(keywords), "entidades_recuperadas": []}
            return empty if return_trace else []

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

        if return_trace:
            return {"keywords": str(keywords), "entidades_recuperadas": entidades}
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
            '{{"uris_aprobadas": ["http://ejemplo/uri1", "http://ejemplo/uri2"]}}. '
            "Si ninguna entidad aplica, devuelve: "
            '{{"uris_aprobadas": []}}.'
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

        uris_aprobadas_set = set()
        max_attempts = 2
        last_error = ""
        last_raw_output = ""

        for attempt in range(1, max_attempts + 1):
            try:
                if attempt == 1:
                    respuesta = self.rag.llm.invoke(prompt_generado)
                else:
                    correction_prompt = (
                        "Tu salida anterior no fue parseable como JSON valido. "
                        f"Error detectado: {last_error}. "
                        "Devuelve exclusivamente un objeto JSON valido con este esquema: "
                        '{"uris_aprobadas": ["http://ejemplo/uri1", "http://ejemplo/uri2"]}. '
                        "Si ninguna aplica, devuelve {'uris_aprobadas': []}. "
                        "No incluyas markdown, comentarios ni texto adicional. "
                        f"Salida anterior: {last_raw_output}"
                    )
                    respuesta = self.rag.llm.invoke(correction_prompt)

                raw_text = respuesta.content if hasattr(respuesta, "content") else str(respuesta)
                last_raw_output = str(raw_text).strip()

                parsed = self._extraer_json_objeto(last_raw_output)
                uris_aprobadas = parsed.get("uris_aprobadas", [])
                if not isinstance(uris_aprobadas, list):
                    raise ValueError("La clave 'uris_aprobadas' no es una lista")

                uris_aprobadas_set = {str(uri).strip() for uri in uris_aprobadas if str(uri).strip()}
                break

            except Exception as exc:
                last_error = str(exc)
                if attempt == max_attempts:
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
            "No devuelvas JSON en esta etapa. "
            "Restriccion de grounding estricta: Solo puedes recomendar ontologias que tengan al menos "
            "un requisito asociado en la matriz JSON. Esta prohibido alucinar o sugerir ontologias externas."
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

    def orquestar_evaluacion(
        self,
        ruta_csv: str,
        max_requirements: Optional[int] = None,
        start_row: Optional[int] = None,
        end_row: Optional[int] = None,
        selected_rows: Optional[List[int]] = None,
        selected_ids: Optional[List[int]] = None,
    ) -> Dict[str, object]:
        """
        Flujo principal Bottom-Up:
        1) Carga requisitos.
        2) Recupera entidades candidatas por requisito.
        3) Filtra entidades con el juez LLM.
        4) Construye TAD requisito->entidades validas.
        5) Genera y muestra matriz de cobertura ontologia->requisitos.
        6) Redacta veredicto final con recomendacion de red de ontologias.
        """
        requisitos = self.cargar_requisitos(
            ruta_csv,
            max_requirements=max_requirements,
            start_row=start_row,
            end_row=end_row,
            selected_rows=selected_rows,
            selected_ids=selected_ids,
        )
        if not requisitos:
            print("[INFO] No hay requisitos para evaluar.")
            return {"tad_requisitos": {}, "matriz_cobertura": {}, "veredicto_final": ""}

        tad_requisitos: Dict[str, List[Dict[str, str]]] = {}
        tad_detallado: Dict[str, Dict[str, Any]] = {}

        for idx, requisito in enumerate(requisitos, start=1):
            print(f"\n[{idx}/{len(requisitos)}] Evaluando requisito: {requisito[:120]}")

            eval_trace = self.evaluar_requisito(requisito, top_k=5, return_trace=True)
            keywords_usadas = eval_trace.get("keywords", "")
            entidades_recuperadas = eval_trace.get("entidades_recuperadas", [])
            entidades_validadas = self.juez_llm(requisito, entidades_recuperadas)

            tad_requisitos[requisito] = entidades_validadas
            tad_detallado[requisito] = {
                "keywords_usadas": keywords_usadas,
                "entidades_recuperadas": entidades_recuperadas,
                "entidades_aprobadas": entidades_validadas,
            }

        matriz_cobertura = self.generar_matriz_cobertura(tad_requisitos)

        print("\n=== Matriz de Cobertura (Ontologia -> Requisitos) ===")
        print(json.dumps(matriz_cobertura, indent=2, ensure_ascii=False))

        veredicto_final = self.redactar_veredicto_final(matriz_cobertura)
        print("\n=== Veredicto Final Recomendado ===")
        print(veredicto_final)

        # Trazas de auditoria
        trace_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trazas_ejecucion.json")
        trace_payload = {
            "ruta_csv": os.path.abspath(ruta_csv),
            "max_requirements": max_requirements,
            "total_requisitos": len(requisitos),
            "tad_requisitos": tad_detallado,
            "matriz_cobertura": matriz_cobertura,
            "veredicto_final": veredicto_final,
        }
        try:
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(trace_payload, f, ensure_ascii=False, indent=2)
            print(f"\n[INFO] Trazas guardadas en: {trace_path}")
        except Exception as exc:
            print(f"\n[WARN] No se pudo guardar trazas_ejecucion.json: {exc}")

        return {
            "tad_requisitos": tad_requisitos,
            "tad_requisitos_detallado": tad_detallado,
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
    evaluador.orquestar_evaluacion(ruta_default, start_row=15, end_row=26)
