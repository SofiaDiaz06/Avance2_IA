import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Cargar datos
df = pd.read_csv('data/proyecto_limpio.csv')

# EVITAR FUGA DE DATOS (DATA LEAKAGE): 
# Seleccionamos entradas que NO definen directamente la etiqueta de la variable objetivo.
# Usamos 'ganancia_unitaria' y 'dias_sin_venta' en lugar de 'unidades_vendidas'.
X = df[['ganancia_unitaria', 'dias_sin_venta']].values
y = df['decisión_abastecer'].values

# Estandarización de características
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_scaled = (X - X_mean) / X_std

# 2. Definición del Perceptrón con Sigmoide y Pérdida Logarítmica
def sigmoide(z):
    # Clip para evitar desbordamientos numéricos en la exponencial
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def calcular_perdida(y_real, y_pred):
    # Binary Cross-Entropy con epsilon para evitar log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_real * np.log(y_pred) + (1 - y_real) * np.log(1 - y_pred))

# Parámetros iniciales
np.random.seed(42)
w = np.random.randn(X_scaled.shape[1]) * 0.01
b = 0.0
tasa_aprendizaje = 0.1
epocas = 100

perdidas_por_epoca = []
m = len(X_scaled)

print("="*60)
print("ENTRENAMIENTO DEL PERCEPTRÓN (SIGMOIDE + PÉRDIDA LOGARÍTMICA)")
print("="*60)

for epoca in range(epocas):
    # Combinación lineal
    z = np.dot(X_scaled, w) + b
    # Predicción con la función sigmoide
    y_pred = sigmoide(z)
    
    # Cálculo de la pérdida logarítmica (Binary Cross-Entropy)
    perdida = calcular_perdida(y, y_pred)
    perdidas_por_epoca.append(perdida)
    
    # Cálculo de gradientes
    dw = (1 / m) * np.dot(X_scaled.T, (y_pred - y))
    db = (1 / m) * np.sum(y_pred - y)
    
    # Actualización de pesos y sesgo por descenso del gradiente
    w -= tasa_aprendizaje * dw
    b -= tasa_aprendizaje * db
    
    if (epoca + 1) % 10 == 0 or epoca == 0:
        print(f"Época {epoca+1:03d} | Pérdida: {perdida:.4f} | w: {w}, b: {b:.4f}")

# Resultados finales y exactitud
z_final = np.dot(X_scaled, w) + b
y_pred_final_prob = sigmoide(z_final)
y_pred_final = (y_pred_final_prob >= 0.5).astype(int)
exactitud = np.mean(y_pred_final == y) * 100

print("\n" + "="*60)
print(f"RESULTADOS FINALES:")
print(f"Pesos finales (w1, w2): {w}")
print(f"Sesgo final (b): {b:.4f}")
print(f"Pérdida inicial: {perdidas_por_epoca[0]:.4f}")
print(f"Pérdida final: {perdidas_por_epoca[-1]:.4f}")
print(f"Exactitud final: {exactitud:.2f}%")
print("="*60)

os.makedirs('.', exist_ok=True)

# 3. Gráfica 1: Curva de Error / Pérdida (Curva de Aprendizaje)
plt.figure(figsize=(7, 4))
plt.plot(range(1, len(perdidas_por_epoca) + 1), perdidas_por_epoca, marker='o', color='crimson', markersize=3)
plt.title('Evolución de la Pérdida Logarítmica (Curva de Aprendizaje)')
plt.xlabel('Época')
plt.ylabel('Pérdida (Binary Cross-Entropy)')
plt.grid(True)
plt.tight_layout()
plt.savefig('grafico_curva_error.png', dpi=150)
plt.close()

# 4. Gráfica 2: Frontera de Decisión
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[y==1, 0], X_scaled[y==1, 1], color='green', label='Abastecer (1)', s=100, edgecolors='k')
plt.scatter(X_scaled[y==0, 0], X_scaled[y==0, 1], color='red', label='No Abastecer / Mantener (0)', s=100, edgecolors='k')

x1_vals = np.linspace(X_scaled[:, 0].min()-0.5, X_scaled[:, 0].max()+0.5, 100)
# Línea de decisión donde w[0]*x1 + w[1]*x2 + b = 0 => x2 = -(w[0]*x1 + b) / w[1]
x2_vals = -(w[0] * x1_vals + b) / w[1]
plt.plot(x1_vals, x2_vals, 'k--', label='Frontera de Decisión (p=0.5)', linewidth=2)

plt.title('Frontera de Decisión del Perceptrón (Sigmoide)')
plt.xlabel('Ganancia Unitaria (Estandarizado)')
plt.ylabel('Días Sin Venta (Estandarizado)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('grafico_frontera_decision.png', dpi=150)
plt.close()
print("\n¡Gráficos de entrenamiento generados y guardados con éxito!")