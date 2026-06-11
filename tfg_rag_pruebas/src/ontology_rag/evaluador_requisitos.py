import csv
import json
import os
import re
import logging
from typing import Any, Dict, List, Optional

from .rag_basico import OntologyRecommender as RAGBasico

logger = logging.getLogger(__name__)


class EvaluadorRequisitos:
    """Flujo de evaluacion Bottom-Up orientado a requisitos funcionales."""

    def __init__(
        self,
        rag: Optional[RAGBasico] = None,
        persist_directory: str = "chroma_db",
        embedding_model: str = "BAAI/bge-m3",
        llm_model: str = "llama3",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        warmup: bool = True,
        output_dir: Optional[str] = None,
    ):
        self.rag = (
            rag
            if rag is not None
            else RAGBasico(
                persist_directory=persist_directory,
                embedding_model=embedding_model,
                llm_model=llm_model,
                reranker_model=reranker_model,
                warmup=warmup,
            )
        )
        self.output_dir = output_dir or os.path.join(os.getcwd(), "resultado")

    @staticmethod
    def _normalizar_requisito(texto: str) -> str:
        if texto is None:
            return ""

        requisito = str(texto).strip()
        requisito = re.sub(r"^\s*\d+\s*[\.;:\-\)]\s*", "", requisito)
        requisito = re.sub(r"\s+", " ", requisito).strip()
        return requisito

    @staticmethod
    def _extraer_json_objeto(texto_llm: str) -> Dict[str, Any]:
        if not texto_llm:
            raise ValueError("Respuesta vacia del LLM")

        cleaned = re.sub(r"^\s*```(?:json)?\s*", "", str(texto_llm).strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.IGNORECASE).strip()

        first_curly = cleaned.find("{")
        last_curly = cleaned.rfind("}")
        if first_curly == -1 or last_curly == -1 or last_curly <= first_curly:
            raise ValueError("No se encontro bloque JSON delimitado por llaves")

        json_candidate = cleaned[first_curly:last_curly + 1].strip()
        
        # 1. Intentar json.loads estándar
        try:
            return json.loads(json_candidate)
        except Exception as e_json:
            logger.warning(f"json.loads fallo: {e_json}. Intentando ast.literal_eval...")

        # 2. Intentar ast.literal_eval para tolerar comillas simples y saltos de línea crudos
        try:
            import ast
            candidate_py = json_candidate.replace("true", "True").replace("false", "False").replace("null", "None")
            parsed = ast.literal_eval(candidate_py)
            if isinstance(parsed, dict):
                return parsed
        except Exception as e_ast:
            logger.warning(f"ast.literal_eval fallo: {e_ast}")

        # 3. Limpieza desesperada de comillas
        try:
            fixed_json = json_candidate
            if "'" in fixed_json and '"' not in fixed_json:
                fixed_json = fixed_json.replace("'", '"')
            return json.loads(fixed_json)
        except Exception as e_final:
            raise ValueError(f"Fallo parsing JSON en todos los intentos. Error original: {e_json}")

    def cargar_requisitos(
        self,
        ruta_csv: str,
        max_requirements: Optional[int] = None,
        start_row: Optional[int] = None,
        end_row: Optional[int] = None,
        selected_rows: Optional[List[int]] = None,
        selected_ids: Optional[List[int]] = None,
    ) -> List[str]:
        if not os.path.exists(ruta_csv):
            logger.error(f"No existe el archivo CSV: {ruta_csv}")
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
            logger.error(f"Fallo leyendo CSV de requisitos: {exc}")
            return []

        if max_requirements is not None and max_requirements > 0:
            return requisitos[:max_requirements]
        return requisitos

    def evaluar_requisito(
        self,
        requisito: str,
        top_k: int = 5,
        return_trace: bool = False,
        retrieval_mode: str = "hybrid",
        use_reranker: bool = True,
    ):
        if not requisito or not requisito.strip():
            return {"keywords": "", "entidades_recuperadas": []} if return_trace else []

        query = requisito.strip()

        try:
            keywords = self.rag.extraction_chain.invoke({"user_request": query})
        except Exception:
            keywords = query

        broad_k = max(top_k * 5, top_k)
        raw_docs = self.rag._hybrid_retrieve(keywords, k=broad_k, retrieval_mode=retrieval_mode)
        if not raw_docs:
            empty = {"keywords": str(keywords), "entidades_recuperadas": [], "raw_retrieved_ontologies": []}
            return empty if return_trace else []

        # Recoger las ontologías de los documentos recuperados originalmente
        raw_ontologies = list(set([doc.metadata.get("source", "unknown") for doc in raw_docs if doc.metadata]))

        if use_reranker:
            pairs = [[query, d.page_content[:500]] for d in raw_docs]
            scores = self.rag.reranker.predict(pairs)
            scored_docs = sorted(zip(raw_docs, scores), key=lambda x: x[1], reverse=True)
            selected = scored_docs[:top_k]
        else:
            selected = [(doc, 1.0) for doc in raw_docs[:top_k]]

        entidades = []
        for doc, _score in selected:
            texto = doc.page_content
            uri = self._extraer_uri(texto)
            ontologia = doc.metadata.get("source", "unknown") if doc.metadata else "unknown"
            entidades.append({"uri": uri, "texto": texto, "ontologia": ontologia})

        if return_trace:
            return {
                "keywords": str(keywords),
                "entidades_recuperadas": entidades,
                "raw_retrieved_ontologies": raw_ontologies,
            }
        return entidades

    def juez_llm(self, requisito: str, entidades_recuperadas: List[Dict[str, str]]) -> List[Dict[str, str]]:
        from langchain_core.prompts import ChatPromptTemplate

        if not requisito or not requisito.strip() or not entidades_recuperadas:
            return []

        entidades_formateadas = []
        for idx, entidad in enumerate(entidades_recuperadas):
            uri = str(entidad.get("uri", "")).strip()
            ontologia = str(entidad.get("ontologia", "unknown")).strip()
            texto = str(entidad.get("texto", "")).strip()
            fragmento = re.sub(r"\s+", " ", texto)[:280]
            entidades_formateadas.append(f"ID: {idx}\nURI: {uri}\nOntologia: {ontologia}\nFragmento: {fragmento}")

        bloque_entidades = "\n\n".join(entidades_formateadas)

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

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt),
        ])

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

        return [entidad for entidad in entidades_recuperadas if str(entidad.get("uri", "")).strip() in uris_aprobadas_set]

    def generar_matriz_cobertura(self, tad_requisitos: Dict[str, List[Dict[str, str]]]) -> Dict[str, List[str]]:
        cobertura: Dict[str, List[str]] = {}

        for requisito, entidades in tad_requisitos.items():
            for entidad in entidades:
                ontologia = entidad.get("ontologia", "unknown")
                if ontologia not in cobertura:
                    cobertura[ontologia] = []

                if requisito not in cobertura[ontologia]:
                    cobertura[ontologia].append(requisito)

        return cobertura

    def redactar_veredicto_final(self, matriz_cobertura: Dict[str, List[str]]) -> Dict[str, Any]:
        from langchain_core.prompts import ChatPromptTemplate

        matriz_json = json.dumps(matriz_cobertura, indent=2, ensure_ascii=False)

        system_prompt = (
            "Actua como un Consultor Experto en Web Semantica y Ontologias.\n"
            "Tu tarea es analizar la matriz de cobertura de requisitos por ontologia y proponer "
            "una red de ontologias recomendada (maximo 3).\n\n"
            "Tu respuesta debe constar de dos partes obligatorias:\n"
            "1. Un reporte detallado en Markdown justificando la recomendacion y analizando la cobertura.\n"
            "2. Al final de tu respuesta, un bloque de codigo JSON estrictamente valido delimitado por ```json y ``` que contenga las listas estructuradas:\n"
            "```json\n"
            "{{\n"
            "  \"recomendadas\": [\"nombre_ontologia1\", \"nombre_ontologia2\"],\n"
            "  \"excluidas\": [\n"
            "    {{\n"
            "      \"ontologia\": \"nombre_ontologia3\",\n"
            "      \"motivo\": \"Explicacion muy breve de por que fue excluida (maximo 15 palabras)\"\n"
            "    }}\n"
            "  ]\n"
            "}}\n"
            "```\n\n"
            "Restricciones:\n"
            "1. Maximo 3 ontologias en 'recomendadas'.\n"
            "2. Solo puedes recomendar o excluir ontologias que tengan al menos un requisito asociado en la matriz JSON.\n"
            "3. En 'excluidas', incluye las ontologias de la matriz de cobertura que no fueron seleccionadas en 'recomendadas'."
        )

        human_prompt = (
            "MATRIZ DE COBERTURA (JSON):\n{matriz_cobertura_json}\n\n"
            "Devuelve el reporte en Markdown seguido por el bloque JSON de recomendadas y excluidas."
        )

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt),
        ])

        prompt_generado = prompt_template.format_messages(matriz_cobertura_json=matriz_json)

        pool_ontologias = list(matriz_cobertura.keys())

        try:
            respuesta = self.rag.llm.invoke(prompt_generado)
            raw_text = respuesta.content if hasattr(respuesta, "content") else str(respuesta)
            
            # Extraer el bloque JSON
            parsed = self._extraer_json_objeto(raw_text)
            
            # Extraer el reporte markdown eliminando el bloque JSON de la respuesta del LLM
            veredicto_markdown = re.sub(r"```json.*?```", "", raw_text, flags=re.DOTALL).strip()
            
            # Validaciones básicas
            if not isinstance(parsed.get("recomendadas"), list):
                parsed["recomendadas"] = []
            if not isinstance(parsed.get("excluidas"), list):
                parsed["excluidas"] = []
            
            parsed["veredicto_markdown"] = veredicto_markdown
            return parsed
        except Exception as exc:
            logger.warning(f"Error generando veredicto estructurado: {exc}. Usando fallback.")
            
            # Fallback en caso de fallo del LLM: ordenar por cobertura (mayor a menor)
            sorted_pool = sorted(pool_ontologias, key=lambda ont: len(matriz_cobertura.get(ont, [])), reverse=True)
            recomendadas = sorted_pool[:3]
            excluidas = []
            for ont in sorted_pool[3:]:
                excluidas.append({
                    "ontologia": ont,
                    "motivo": "Excluida automáticamente debido a límite de recomendación y menor cobertura."
                })
                
            fallback_markdown = (
                "### Veredicto Final (Fallback)\n\n"
                f"No se pudo estructurar el veredicto debido a un error: {exc}.\n\n"
                "**Ontologías Recomendadas en Red:**\n" + 
                "\n".join([f"- **{ont}**" for ont in recomendadas])
            )
            
            return {
                "recomendadas": recomendadas,
                "excluidas": excluidas,
                "veredicto_markdown": fallback_markdown
            }

    def orquestar_evaluacion(
        self,
        ruta_csv: str,
        max_requirements: Optional[int] = None,
        start_row: Optional[int] = None,
        end_row: Optional[int] = None,
        selected_rows: Optional[List[int]] = None,
        selected_ids: Optional[List[int]] = None,
        retrieval_mode: str = "hybrid",
        use_reranker: bool = True,
    ) -> Dict[str, object]:
        requisitos = self.cargar_requisitos(
            ruta_csv,
            max_requirements=max_requirements,
            start_row=start_row,
            end_row=end_row,
            selected_rows=selected_rows,
            selected_ids=selected_ids,
        )
        if not requisitos:
            logger.info("No hay requisitos para evaluar.")
            return {"tad_requisitos": {}, "matriz_cobertura": {}, "veredicto_final": ""}

        tad_requisitos: Dict[str, List[Dict[str, str]]] = {}
        tad_detallado: Dict[str, Dict[str, Any]] = {}

        for idx, requisito in enumerate(requisitos, start=1):
            logger.info(f"[{idx}/{len(requisitos)}] Evaluando requisito: {requisito[:120]}")

            eval_trace = self.evaluar_requisito(
                requisito,
                top_k=5,
                return_trace=True,
                retrieval_mode=retrieval_mode,
                use_reranker=use_reranker,
            )
            keywords_usadas = eval_trace.get("keywords", "")
            entidades_recuperadas = eval_trace.get("entidades_recuperadas", [])
            raw_retrieved_ontologies = eval_trace.get("raw_retrieved_ontologies", [])
            entidades_validadas = self.juez_llm(requisito, entidades_recuperadas)

            tad_requisitos[requisito] = entidades_validadas
            tad_detallado[requisito] = {
                "keywords_usadas": keywords_usadas,
                "entidades_recuperadas": entidades_recuperadas,
                "raw_retrieved_ontologies": raw_retrieved_ontologies,
                "entidades_aprobadas": entidades_validadas,
            }

        matriz_cobertura = self.generar_matriz_cobertura(tad_requisitos)

        logger.info("=== Matriz de Cobertura (Ontologia -> Requisitos) ===")
        logger.info(json.dumps(matriz_cobertura, indent=2, ensure_ascii=False))

        veredicto_estructurado = self.redactar_veredicto_final(matriz_cobertura)
        veredicto_final_markdown = veredicto_estructurado.get("veredicto_markdown", "")
        
        logger.info("=== Veredicto Final Recomendado ===")
        logger.info(veredicto_final_markdown)

        # Crear carpeta resultado
        os.makedirs(self.output_dir, exist_ok=True)
        trace_path = os.path.join(self.output_dir, "trazas_ejecucion.json")

        trace_payload = {
            "ruta_csv": os.path.abspath(ruta_csv),
            "max_requirements": max_requirements,
            "total_requisitos": len(requisitos),
            "tad_requisitos": tad_detallado,
            "matriz_cobertura": matriz_cobertura,
            "veredicto_final": veredicto_final_markdown,
            "veredicto_estructurado": veredicto_estructurado,
        }
        try:
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(trace_payload, f, ensure_ascii=False, indent=2)
            logger.info(f"Trazas guardadas en: {trace_path}")
        except Exception as exc:
            logger.warning(f"No se pudo guardar trazas_ejecucion.json: {exc}")

        return {
            "tad_requisitos": tad_requisitos,
            "tad_requisitos_detallado": tad_detallado,
            "matriz_cobertura": matriz_cobertura,
            "veredicto_final": veredicto_final_markdown,
            "veredicto_estructurado": veredicto_estructurado,
        }

    @staticmethod
    def _extraer_uri(texto_documento: str) -> str:
        if not texto_documento:
            return ""

        match = re.search(r"^URI:\s*(\S+)", texto_documento, flags=re.MULTILINE)
        if match:
            return match.group(1)
        return ""

