import pandas as pd
import numpy as np
import io
import os

# 1. Cargar el dataset simulado
csv_data = """id_producto,nombre,categoria,precio_venta,costo_compra,unidades_vendidas,stock_actual,dias_sin_venta,calificacion_cliente
1,Organizador_Plástico,Hogar,15000,8000,145,18,2,4.7
2,Vaso_Plástico_6_Unidades,Hogar,12000,6500,98,25,4,4.5
3,Recipiente_Hermético,Hogar,18000,10000,76,14,6,4.6
4,Espejo_Pequeño,Belleza,12000,5500,112,20,3,4.8
5,Labial_Mate,Belleza,10000,4500,185,22,1,4.7
6,Esmalte_Para_Uñas,Belleza,7000,3000,210,30,1,4.6
7,Brocha_Maquillaje,Belleza,9000,4000,125,15,3,4.5
8,Cuaderno_Argollado,Papeleria,12000,6500,160,35,2,4.8
9,Paquete_Lapiceros,Papeleria,6000,2800,245,40,1,4.7
10,Marcadores_Colores,Papeleria,15000,8000,88,20,5,4.4
11,Carpeta_Plástica,Papeleria,4000,1800,190,45,2,4.6
12,Globo_Número,Fiesta,5000,2200,135,25,3,4.7
13,Bolsas_Regalo,Fiesta,4000,1800,175,32,2,4.6
14,Vela_Aromática,Fiesta,10000,5500,42,38,18,4.1
15,Confeti_Decorativo,Fiesta,5000,2500,28,55,35,3.9
16,Bolsito_Casual,Accesorios,25000,12000,65,18,7,4.5
17,Monedero,Accesorios,12000,5500,92,12,4,4.6
18,Llavero_Decorativo,Accesorios,6000,2500,155,20,2,4.7
19,Carrito_Juguete,Juguetes,15000,8500,34,48,22,4.0
20,Muñeca_Pequeña,Juguetes,18000,10000,25,42,30,3.8
21,Rompecabezas,Juguetes,20000,11000,18,35,45,3.7
22,Pelota_Pequeña,Juguetes,8000,3500,120,15,5,4.4
"""

df = pd.read_csv(io.StringIO(csv_data))

print("="*60)
print("INSPECCIÓN INICIAL DE DATOS")
print("="*60)
print(f"Dimensiones (shape): {df.shape}")
print(f"Valores nulos por columna:\n{df.isnull().sum()}")
print(f"Filas duplicadas: {df.duplicated().sum()}")

# Corrige erratas de categoría
df['categoria'] = df['categoria'].replace({'Juquetes': 'Juguetes'})

# 2. Creación de variables derivadas clave
df['ganancia_unitaria'] = df['precio_venta'] - df['costo_compra']
df['margen_ganancia'] = df['ganancia_unitaria'] / df['precio_venta']
df['rotacion_diaria'] = df['unidades_vendidas'] / (df['dias_sin_venta'] + 1)

# 3. Definición de la Variable Objetivo (0 o 1)
df['decisión_abastecer'] = np.where((df['unidades_vendidas'] >= 100) & (df['margen_ganancia'] >= 0.40), 1, 0)

print("\n" + "="*60)
print("INSPECCIÓN FINAL DESPUÉS DE LA LIMPIEZA")
print("="*60)
print(f"Dimensiones finales: {df.shape}")
print(f"Nulos totales: {df.isnull().sum().sum()}")

# Exportar datos limpios
os.makedirs('data', exist_ok=True)
df.to_csv('data/proyecto_limpio.csv', index=False)
print("Dataset limpio guardado exitosamente en 'data/proyecto_limpio.csv'.")