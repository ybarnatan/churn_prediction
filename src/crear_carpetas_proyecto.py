import os
import logging

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import os
import logging
from .config import DIRS_TO_CREATE, LOGS_DIR # Importa la lista de carpetas y una variable de ejemplo para el log

# Configuración básica de logging
# Nota: La configuración principal del logger se hará en main.py, esto es para que el script funcione de forma independiente si es necesario.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


def crear_carpetas_proyecto():
    """
    Crea la estructura de carpetas necesaria en el directorio raíz del proyecto 
    (al mismo nivel que main.py y conf.yaml) si no existen.
    
    Las carpetas creadas se definen en la constante DIRS_TO_CREATE de src.config 
    (obtenidas de conf.yaml).
    
    El directorio raíz se asume dos niveles arriba de este script (src/crear_carpetas_proyecto.py).
    """
    
    # El directorio raíz (donde están main.py y conf.yaml) es dos niveles arriba de src/
    ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
    
    # La lista de carpetas a crear se obtiene de config.py (leído de conf.yaml)
    carpetas_necesarias = DIRS_TO_CREATE
    
    # Eliminamos 'src' de la lista para evitar errores si ya estamos dentro de 'src' 
    # y la creamos por accidente. 'src' ya debe existir.
    if 'src' in carpetas_necesarias:
        carpetas_necesarias.remove('src')
    
    logger.info(f"Comenzando la creación de carpetas en: {os.path.abspath(ROOT_DIR)}")
    
    for folder in carpetas_necesarias:
        # Construye la ruta completa de la carpeta
        folder_path = os.path.join(ROOT_DIR, folder)
        
        # Verifica si existe y la crea si no
        if not os.path.exists(folder_path):
            try:
                os.makedirs(folder_path)
                logger.info(f"✅ Carpeta creada: '{folder}'")
            except Exception as e:
                logger.error(f"❌ Error al crear la carpeta '{folder}': {e}")
        else:
            logger.debug(f"ℹ️ Carpeta ya existe: '{folder}'")

    # Mover el archivo conf.yaml a un directorio de configuración no es necesario en este caso
    # porque la función se llama desde main.py.