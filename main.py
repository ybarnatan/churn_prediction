import pandas as pd
import os
import datetime
import logging
from src.dataloader import cargar_datos


#Log para el main
os.makedirs("logs", exist_ok=True) #Corroboro que exista la carpeta logs
fecha = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
nombre = f"log_fecha_{fecha}.log"
path = "C:/Users/ybbar/OneDrive/Desktop/DMEyF2025_Competencia01_Proyect_Wedesnday/data/competencia_01_crudo.csv"


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/{nombre}", mode="w", encoding="utf-8"),
        logging.StreamHandler()
        ])

logger = logging.getLogger(__name__)



def main():
    print("Hola Mundo desde main!")
    logger.info("Inicio de ejecucion del programa")
    df = cargar_datos(path)
    print(df.head(4))
    logger.info(f"Fin de ejecucion del programa. Revisar detalle en logs/{nombre}")

if __name__ == "__main__": # Asegura que solo se ejecute main() si corremos "python main.py" en terminal, pero no ejecute main() si lo importamos en otro archivo.
    main()