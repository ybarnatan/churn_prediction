import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_json("resultados/lgb_optimization_competencia01_iteraciones.json")
print(df.head())

#Graficar ganancia Optuna eje x trial_number eje y ganancia
# Configurar el estilo
plt.style.use('seaborn-v0_8')
plt.figure(figsize=(12, 6))

# Crear el gráfico
sns.lineplot(data=df, x='trial_number', y='value', marker='o', linewidth=2, markersize=6)

# Personalizar el gráfico
plt.title('Evolución de la Ganancia por Iteración de Optuna', fontsize=16, fontweight='bold')
plt.xlabel('Número de Trial', fontsize=12)
plt.ylabel('Ganancia', fontsize=12)
plt.grid(True, alpha=0.3)

# Formatear el eje y para mostrar valores en millones
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))

# Ajustar layout
plt.tight_layout()

# Mostrar el gráfico
plt.show()

# Opcional: Mostrar estadísticas básicas
print(f"\nEstadísticas de ganancia:")
print(f"Ganancia máxima: {df['value'].max():,.0f}")
print(f"Ganancia mínima: {df['value'].min():,.0f}")
print(f"Ganancia promedio: {df['value'].mean():,.0f}")
print(f"Trial con mejor ganancia: {df.loc[df['value'].idxmax(), 'trial_number']}")