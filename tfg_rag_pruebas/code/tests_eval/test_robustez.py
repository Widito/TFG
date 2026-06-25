import os
import sys
import shutil
import logging
import unittest
from pathlib import Path

# Configurar sys.path para importar localmente desde src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, os.path.join(project_root, "src"))

from ontology_rag import OntologyIndexer, OntologyRecommender
from langchain_core.language_models.chat_models import BaseChatModel

# Configurar logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("TestRobustez")


class BrokenLLM(BaseChatModel):
    """
    Clase Mock para simular un LLM (Ollama) desconectado / inaccesible.
    Lanza una excepción de conexión inmediatamente cuando es invocado.
    """
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise ConnectionError(
            "Simulated Ollama Connection Refused (port 11434 unreachable)."
        )

    @property
    def _llm_type(self) -> str:
        return "broken-ollama"


class TestRobustez(unittest.TestCase):
    
    def setUp(self):
        self.temp_source_dir = os.path.join(project_root, "test_temp_source")
        self.temp_persist_dir = os.path.join(project_root, "chroma_db_test_robustez")
        
        # Limpieza previa por seguridad
        self._cleanup()
        
        # Crear directorio temporal para ontologías de origen
        os.makedirs(self.temp_source_dir, exist_ok=True)

    def tearDown(self):
        self._cleanup()

    def _cleanup(self):
        import gc
        import time
        
        # Forzar recolección de basura para cerrar descriptores de archivos de sqlite/chroma
        gc.collect()
        time.sleep(0.3)
        
        for path in [self.temp_source_dir, self.temp_persist_dir]:
            if os.path.exists(path):
                for attempt in range(5):
                    try:
                        shutil.rmtree(path)
                        break
                    except PermissionError:
                        gc.collect()
                        time.sleep(0.3)
                    except Exception as e:
                        logger.warning(f"No se pudo eliminar {path}: {e}")
                        break


    def test_ingesta_rdf_corrupto(self):
        logger.info("\n=== [TEST] Ingesta de RDF Corrupto ===")
        
        # 1. Obtener ontología real y válida (facility.ttl es ligera)
        real_ontology_path = os.path.join(project_root, "dataset", "facility.ttl")
        self.assertTrue(
            os.path.exists(real_ontology_path), 
            f"No se encuentra la ontología de referencia en {real_ontology_path}"
        )
        
        # Copiar ontología válida al directorio temporal
        dest_valid_path = os.path.join(self.temp_source_dir, "facility.ttl")
        shutil.copy2(real_ontology_path, dest_valid_path)
        logger.info(f"Copiada ontología válida a {dest_valid_path}")
        
        # 2. Copiar archivo corrupto elaborado desde el dataset
        src_corrupt_path = os.path.join(project_root, "dataset", "corrupto.ttl")
        self.assertTrue(
            os.path.exists(src_corrupt_path),
            f"No se encuentra el archivo corrupto elaborado en {src_corrupt_path}"
        )
        dest_corrupt_path = os.path.join(self.temp_source_dir, "corrupto.ttl")
        shutil.copy2(src_corrupt_path, dest_corrupt_path)
        logger.info(f"Copiado archivo corrupto elaborado a {dest_corrupt_path}")
        
        # 3. Inicializar indexador con persistencia aislada
        indexer = OntologyIndexer(
            source_directories=[self.temp_source_dir],
            persist_directory=self.temp_persist_dir,
            recreate_db=True,
            embedding_model="BAAI/bge-m3"
        )
        
        # 4. Ejecutar indexación y verificar que no crashee
        logger.info("Ejecutando build_index()... (Debe avisar por logs y continuar)")
        try:
            indexer.build_index()
            logger.info(" build_index() completado sin lanzar excepciones.")
        except Exception as e:
            self.fail(f"build_index() lanzó una excepción inesperada: {e}")
            
        # 5. Verificar que los vectores de facility.ttl existen y corrupto.ttl fue recuperado
        vs = indexer.get_vectorstore()
        collection = vs._collection
        count = collection.count()
        logger.info(f"Número de fragmentos indexados en ChromaDB: {count}")
        
        self.assertGreater(count, 0, "No se indexó ningún fragmento válido.")
        
        # Buscar las fuentes indexadas
        results = collection.get()
        metadatas = results.get("metadatas", [])
        sources = {m.get("source") for m in metadatas if m}
        uris = {m.get("uri") for m in metadatas if m}
        
        logger.info(f"Fuentes indexadas encontradas en BD: {sources}")
        logger.info(f"URIs encontradas en BD: {uris}")
        
        self.assertIn("facility.ttl", sources, "facility.ttl debería estar en la base de datos.")
        self.assertIn("corrupto.ttl", sources, "corrupto.ttl debería estar indexado tras recuperarse.")
        
        # Verificar que se recuperaron las clases válidas específicas de corrupto.ttl
        self.assertIn("http://example.org/test#ValidZone", uris, "ValidZone no fue recuperada de corrupto.ttl")
        self.assertIn("http://example.org/test#RecoveredSensor", uris, "RecoveredSensor no fue recuperada de corrupto.ttl")
        logger.info(" Test de ingesta de RDF corrupto superado con éxito.")
        
        # Liberar referencias de base de datos para Windows
        vs = None
        collection = None
        indexer = None


    def test_ollama_offline(self):
        logger.info("\n=== [TEST] Ollama Offline (Simulación de Fallo de Red) ===")
        
        # Inyectar el LLM roto al recomendador
        broken_llm = BrokenLLM()
        
        # Usar la base de datos de producción real si existe para la inicialización (solo lectura)
        # O si no existe, usar la temporal para evitar errores de BD vacía
        prod_db_path = os.path.join(project_root, "chroma_db")
        db_path = prod_db_path if os.path.exists(prod_db_path) else self.temp_persist_dir
        
        # Si no hay DB para el BM25, creamos una mínima indexando primero
        if not os.path.exists(db_path) or len(os.listdir(db_path)) == 0:
            logger.info("Creando base de datos temporal mínima para el test de Ollama...")
            real_ontology_path = os.path.join(project_root, "dataset", "facility.ttl")
            shutil.copy2(real_ontology_path, os.path.join(self.temp_source_dir, "facility.ttl"))
            indexer = OntologyIndexer(
                source_directories=[self.temp_source_dir],
                persist_directory=self.temp_persist_dir,
                recreate_db=True
            )
            indexer.build_index()
            db_path = self.temp_persist_dir
            
        logger.info(f"Inicializando recomendador con BD: {db_path} e inyectando BrokenLLM...")
        recommender = OntologyRecommender(
            persist_directory=db_path,
            llm=broken_llm,
            warmup=True # Esto lanzará warning pero no debe crashear
        )
        
        # Ejecutar pipeline y verificar que propague la excepción controladamente en lugar de colgarse
        query = "Zones are areas with spatial 3D volumes"
        logger.info(f"Ejecutando run_pipeline con query: '{query}'")
        
        with self.assertRaises(ConnectionError) as context:
            recommender.run_pipeline(query)
            
        logger.info(f" Excepción capturada correctamente: {context.exception}")
        self.assertIn("Simulated Ollama Connection Refused", str(context.exception))
        logger.info(" Test de Ollama Offline superado con éxito.")
        
        # Liberar referencias de base de datos para Windows
        recommender = None
        if 'indexer' in locals():
            indexer = None


    def test_entradas_limite(self):
        logger.info("\n=== [TEST] Entradas Límite y Queries Fuera de Dominio ===")
        
        prod_db_path = os.path.join(project_root, "chroma_db")
        db_path = prod_db_path if os.path.exists(prod_db_path) else self.temp_persist_dir
        
        # Asegurar base de datos mínima
        if not os.path.exists(db_path) or len(os.listdir(db_path)) == 0:
            logger.info("Creando base de datos temporal mínima para test de entradas límite...")
            real_ontology_path = os.path.join(project_root, "dataset", "facility.ttl")
            shutil.copy2(real_ontology_path, os.path.join(self.temp_source_dir, "facility.ttl"))
            indexer = OntologyIndexer(
                source_directories=[self.temp_source_dir],
                persist_directory=self.temp_persist_dir,
                recreate_db=True
            )
            indexer.build_index()
            db_path = self.temp_persist_dir

        # Nota: Usamos un mock LLM que responde sin fallos para evaluar el pipeline con queries atípicas
        class DummyLLM(BaseChatModel):
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                # Retornar respuesta simulada (vacía o sin ontología útil)
                from langchain_core.outputs import ChatGeneration, ChatResult
                from langchain_core.messages import AIMessage
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="""{
                    "RAZONAMIENTO": "La petición está vacía o no tiene relación con ontologías de edificación/sensores.",
                    "ONTOLOGÍA_RECOMENDADA": "ninguna"
                }"""))])
            
            @property
            def _llm_type(self) -> str:
                return "dummy-llm"

        recommender = OntologyRecommender(
            persist_directory=db_path,
            llm=DummyLLM(),
            warmup=False
        )

        # 1. Query vacía
        logger.info("Probando query vacía...")
        try:
            res_vacia = recommender.run_pipeline("")
            logger.info(f"Resultado query vacía: {res_vacia}")
            self.assertIsNotNone(res_vacia)
        except Exception as e:
            self.fail(f"El RAG falló con query vacía: {e}")

        # 2. Query fuera de dominio
        logger.info("Probando query fuera de dominio ('cómo hacer una tarta de manzana')...")
        try:
            res_od = recommender.run_pipeline("cómo hacer una tarta de manzana")
            logger.info(f"Resultado fuera de dominio: {res_od}")
            # Verificamos que no se recomiende nada o que se retorne una respuesta controlada
            self.assertIsNotNone(res_od)
        except Exception as e:
            self.fail(f"El RAG falló con query fuera de dominio: {e}")

        logger.info("Test de entradas límite superado con éxito.")
        
        # Liberar referencias de base de datos para Windows
        recommender = None
        if 'indexer' in locals():
            indexer = None



if __name__ == "__main__":
    unittest.main()
