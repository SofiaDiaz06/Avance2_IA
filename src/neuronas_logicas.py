import matplotlib.pyplot as plt

# ==========================================
# FUNCIONES COMUNES DE LA NEURONA ARTIFICIAL
# ==========================================

def escalon(z):
    """Función de activación escalón."""
    return 1 if z >= 0 else 0

def entrenar_neurona(datos, w1, w2, b, tasa_aprendizaje, nombre_puerta, max_epocas=20):
    """
    Entrena una neurona artificial para una compuerta lógica dada.
    Retorna la lista de errores por época y los parámetros finales.
    """
    print("=" * 60)
    print(f"ENTRENAMIENTO DE NEURONA PARA COMPUERTA: {nombre_puerta}")
    print("=" * 60)
    print(f"Parámetros iniciales -> w1: {w1}, w2: {w2}, b: {b}, Tasa: {tasa_aprendizaje}\n")
    
    errores_por_epoca = []
    
    for epoca in range(max_epocas):
        error_total = 0
        print(f"--- Época {epoca + 1} ---")
        
        for entrada in datos:
            # Soportar compuertas de 2 entradas (AND/OR) o de 1 entrada (NOT)
            if len(entrada) == 3:
                x1, x2, y_real = entrada
                # Para compuerta NOT simulamos que x2 no afecta usando w2=0 o ignorándolo
                z = (w1 * x1) + (w2 * x2) + b
            else:
                x1, y_real = entrada
                x2 = 0 # No se usa en NOT
                z = (w1 * x1) + b
                
            y_pred = escalon(z)
            error = y_real - y_pred
            error_total += abs(error)
            
            # Ajuste de pesos y bias (Descenso del gradiente simplificado)
            w1 += tasa_aprendizaje * error * x1
            if len(entrada) == 3:
                w2 += tasa_aprendizaje * error * x2
            b += tasa_aprendizaje * error
            
            if len(entrada) == 3:
                print(f" Entrada: ({x1}, {x2}) | Real: {y_real} | Pred: {y_pred} | Error: {error}")
            else:
                print(f" Entrada: ({x1}) | Real: {y_real} | Pred: {y_pred} | Error: {error}")
                
        errores_por_epoca.append(error_total)
        print(f" Error total en época {epoca + 1}: {error_total}")
        print(f" Parámetros actuales -> w1: {w1:.3f}, w2: {w2:.3f}, b: {b:.3f}\n")
        
        if error_total == 0:
            print(f"¡La neurona aprendió la compuerta {nombre_puerta} en {epoca + 1} épocas!\n")
            break
            
    return w1, w2, b, errores_por_epoca

# ==========================================
# 1. COMPUERTA OR
# ==========================================
# Datos OR: (x1, x2, y_real) -> Devuelve 1 si al menos una entrada es 1
datos_or = [
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 1)
]

w1_or, w2_or, b_or = 0.2, 0.2, -0.1
w1_or, w2_or, b_or, errores_or = entrenar_neurona(datos_or, w1_or, w2_or, b_or, 0.1, "OR")

# Graficar error OR
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(errores_or) + 1), errores_or, marker='o', color='blue', linewidth=2)
plt.title('Evolución del Error - Compuerta OR')
plt.xlabel('Época')
plt.ylabel('Error Total')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('evolucion_error_or.png', dpi=150)
plt.close()
print("Gráfico guardado: evolucion_error_or.png\n")


# ==========================================
# 2. COMPUERTA NOT
# ==========================================
# Datos NOT: (x1, y_real) -> Invierte el valor (0->1, 1->0)
datos_not = [
    (0, 1),
    (1, 0)
]

# Para la compuerta NOT solo necesitamos w1 y b
w1_not, w2_not, b_not = -0.5, 0.0, 0.3
print("=" * 60)
print("ENTRENAMIENTO DE NEURONA PARA COMPUERTA: NOT")
print("=" * 60)
print(f"Parámetros iniciales -> w1: {w1_not}, b: {b_not}\n")

errores_not = []
for epoca in range(10):
    error_total = 0
    print(f"--- Época {epoca + 1} ---")
    for x1, y_real in datos_not:
        z = (w1_not * x1) + b_not
        y_pred = escalon(z)
        error = y_real - y_pred
        error_total += abs(error)
        
        # Ajuste
        w1_not += 0.1 * error * x1
        b_not += 0.1 * error
        print(f" Entrada: ({x1}) | Real: {y_real} | Pred: {y_pred} | Error: {error}")
        
    errores_not.append(error_total)
    print(f" Error total: {error_total}")
    print(f" Parámetros actuales -> w1: {w1_not:.3f}, b: {b_not:.3f}\n")
    if error_total == 0:
        print(f"¡La neurona aprendió la compuerta NOT en {epoca + 1} épocas!\n")
        break

# Graficar error NOT
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(errores_not) + 1), errores_not, marker='s', color='green', linewidth=2)
plt.title('Evolución del Error - Compuerta NOT')
plt.xlabel('Época')
plt.ylabel('Error Total')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('evolucion_error_not.png', dpi=150)
plt.close()
print("Gráfico guardado: evolucion_error_not.png\n")
print("¡Entrenamiento y pruebas de compuertas lógicas finalizadas con éxito!")