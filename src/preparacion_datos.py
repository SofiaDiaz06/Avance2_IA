import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", context="talk")

def cargar_datos(archivo_csv):
    try:
        df = pd.read_csv(archivo_csv)
        print(f"Datos cargados correctamente: {df.shape[0]} filas, {df.shape[1]} columnas.")
        return df
    except FileNotFoundError:
        print(f"Error: El archivo '{archivo_csv}' no existe.")
        return None

def explorar_datos(df):
    print("\nPRIMERAS 5 FILAS:")
    print(df.head())
    print("\nINFORMACIÓN GENERAL:")
    print(df.info())
    print("\nESTADÍSTICAS DESCRIPTIVAS:")
    print(df.describe())
    print("\nVALORES NULOS POR COLUMNA:")
    print(df.isnull().sum())

def limpiar_datos(df):
    df_limpio = df.copy()
    mediana_edad = df_limpio['cliente_edad'].median()
    df_limpio['cliente_edad'] = df_limpio['cliente_edad'].fillna(mediana_edad)
    print(f"\nValores nulos en 'cliente_edad' imputados con la mediana: {mediana_edad}")
    df_limpio['total_venta'] = df_limpio['precio_unitario'] * df_limpio['cantidad']
    print("Columna 'total_venta' creada.")
    return df_limpio

def visualizar_datos(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='total_venta', bins=10, kde=True, color='skyblue')
    plt.title('Distribución del Total de Ventas')
    plt.xlabel('Total de Venta ($)')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.savefig('distribucion_ventas.png', dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='categoria', y='total_venta', estimator=np.sum, errorbar=None, palette='viridis')
    plt.title('Total de Ventas por Categoría')
    plt.xlabel('Categoría')
    plt.ylabel('Total de Ventas ($)')
    plt.tight_layout()
    plt.savefig('ventas_por_categoria.png', dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='cliente_edad', y='total_venta', hue='categoria', style='metodo_pago', s=100)
    plt.title('Relación: Edad del Cliente vs Total de Venta')
    plt.xlabel('Edad del Cliente')
    plt.ylabel('Total de Venta ($)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('edad_vs_venta.png', dpi=150)
    plt.close()

def preparar_ml(df):
    df_ml = df.copy()
    df_ml = pd.get_dummies(df_ml, columns=['categoria', 'metodo_pago'], prefix=['cat', 'pago'])
    print("\nCodificación One-Hot aplicada.")
    columnas_a_eliminar = ['id_venta', 'fecha', 'producto', 'ciudad']
    df_ml = df_ml.drop(columns=columnas_a_eliminar, errors='ignore')
    print("Columnas no numéricas eliminadas.")
    print("\nDATAFRAME LISTO PARA MACHINE LEARNING (primeras 5 filas):")
    print(df_ml.head())
    return df_ml

def main():
    print("=" * 60)
    print("PREPARACIÓN DE DATOS PARA MACHINE LEARNING - VENTAS")
    print("=" * 60)
    df = cargar_datos('ventas_tienda.csv')
    if df is None:
        return
    explorar_datos(df)
    df_limpio = limpiar_datos(df)
    visualizar_datos(df_limpio)
    df_ml = preparar_ml(df_limpio)
    df_ml.to_csv('ventas_preparadas_ml.csv', index=False)
    print("\nDataset preparado guardado como 'ventas_preparadas_ml.csv'")
    print("\nProceso completado exitosamente.")

if __name__ == "__main__":
    main()
