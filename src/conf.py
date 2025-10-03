# =====================
# Configuracion global
# =====================
SEMILLA = 811157
MES_TRAIN = 202102 #Entreno la Bayesiana
MES_VALIDACION = 202103 # Valido la Bayesiana
MES_TEST = 202104 #Test final antes de predecir
GANANCIA_ACIERTO = 780000
COSTO_ESTIMULO = 20000



#Traigo columnas "C" (conteo ), "M" (monto) o "O" (otro)
import src.identify_traits as Traits
vars_tipo = Traits.trait_type()