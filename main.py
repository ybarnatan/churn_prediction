def main():
    print("Hola mundo desde main!")

if __name__ == "__main__": # Asegura que solo se ejecute main() si corremos "python main.py" en terminal, pero no ejecute main() si lo importamos en otro archivo.
    main()
import pandas as pd
import os
import datetime
import logging


#Log para el main
os.makedirs("logs", exist_ok=True) #Corroboro que exista la carpeta logs
fecha = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
nombre = f"logs/log_fecha_{fecha}.log"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/{nombre}", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)



def main():
    logger.info("Inicio de ejecucion del programa")
    print("Hola mundo desde main! ... ")
    
    
    logger.info("Fin de ejecucion del programa")

if __name__ == "__main__": # Asegura que solo se ejecute main() si corremos "python main.py" en terminal, pero no ejecute main() si lo importamos en otro archivo.
    main()