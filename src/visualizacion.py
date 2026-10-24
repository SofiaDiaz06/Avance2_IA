import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/proyecto_limpio.csv')
sns.set_theme(style="whitegrid")

# Gráfico 1: Histograma de Rotación de Ventas
plt.figure(figsize=(8, 5))
sns.histplot(df['unidades_vendidas'], kde=True, color='teal', bins=8)
plt.title('Distribución de Unidades Vendidas en el Mes')
plt.xlabel('Unidades Vendidas')
plt.ylabel('Cantidad de Productos')
plt.tight_layout()
plt.savefig('grafico_1_distribucion_ventas.png', dpi=150)
plt.close()

# Gráfico 2: Ventas Promedio por Categoría
plt.figure(figsize=(9, 5))
sns.barplot(data=df, x='categoria', y='unidades_vendidas', hue='categoria', errorbar=None, palette='viridis', legend=False)
plt.title('Promedio de Unidades Vendidas por Categoría')
plt.xlabel('Categoría de Productos')
plt.ylabel('Unidades Vendidas Promedio')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('grafico_2_ventas_por_categoria.png', dpi=150)
plt.close()

# Gráfico 3: Dispersión - Entradas vs Variable Objetivo
plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df, 
    x='unidades_vendidas', 
    y='ganancia_unitaria', 
    hue='decisión_abastecer', 
    style='decisión_abastecer',
    palette={1: 'green', 0: 'red'},
    s=120
)
plt.title('Relación: Ventas vs Ganancia Unitaria y Decisión de Abastecimiento')
plt.xlabel('Unidades Vendidas')
plt.ylabel('Ganancia Unitaria ($)')
plt.tight_layout()
plt.savefig('grafico_3_dispersion_objetivo.png', dpi=150)
plt.close()

print("¡Gráficos generados y guardados correctamente sin errores!")