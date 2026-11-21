"""
Entrena un clasificador Random Forest para reproducir el nivel experimental
asignado por el motor matemático del proyecto.

Este modelo aprende las etiquetas experimentales del algoritmo existente.
No constituye un modelo clínico validado ni sustituye el motor matemático.
"""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent
ARCHIVO_DATASET = BASE_DIR / "data" / "dataset_entrenamiento_ia.csv"
CARPETA_RESULTADOS = BASE_DIR / "results"

VARIABLES_IA = [
    "entrada_spo2",
    "entrada_pa",
    "entrada_frecuencia_cardiaca",
    "entrada_frecuencia_respiratoria",
    "entrada_temperatura",
    "entrada_dolor",
    "entrada_edad",
]

OBJETIVO = "nivel_experimental"
ORDEN_NIVELES = ["Nivel I", "Nivel II", "Nivel III", "Nivel IV", "Nivel V"]


def cargar_dataset():
    if not ARCHIVO_DATASET.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset: {ARCHIVO_DATASET}\n"
            "Ejecuta primero: python generar_dataset_ia.py"
        )

    df = pd.read_csv(ARCHIVO_DATASET)

    columnas_necesarias = VARIABLES_IA + [OBJETIVO]
    faltantes = [col for col in columnas_necesarias if col not in df.columns]

    if faltantes:
        raise ValueError(f"Faltan columnas necesarias en el dataset: {faltantes}")

    df = df[columnas_necesarias].copy()
    df = df.dropna(subset=columnas_necesarias)

    if df.empty:
        raise ValueError("El dataset no contiene registros completos para entrenar.")

    niveles_desconocidos = sorted(set(df[OBJETIVO]) - set(ORDEN_NIVELES))
    if niveles_desconocidos:
        raise ValueError(f"Se encontraron niveles no reconocidos: {niveles_desconocidos}")

    return df


def entrenar_modelo():
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    df = cargar_dataset()
    X = df[VARIABLES_IA]
    y = df[OBJETIVO]

    conteo = y.value_counts()
    clases_con_menos_de_dos = conteo[conteo < 2]

    if not clases_con_menos_de_dos.empty:
        raise ValueError(
            "No se puede realizar una división estratificada: "
            f"hay clases con menos de dos registros: {clases_con_menos_de_dos.to_dict()}"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    modelo = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    modelo.fit(X_train, y_train)

    predicciones = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, predicciones)

    etiquetas_presentes = [
        nivel for nivel in ORDEN_NIVELES
        if nivel in set(y_test) or nivel in set(predicciones)
    ]

    reporte = classification_report(
        y_test,
        predicciones,
        labels=etiquetas_presentes,
        output_dict=True,
        zero_division=0,
    )

    matriz = confusion_matrix(
        y_test,
        predicciones,
        labels=etiquetas_presentes,
    )

    # Guardar el modelo y los artefactos de evaluación.
    ruta_modelo = CARPETA_RESULTADOS / "modelo_ia_random_forest.joblib"
    joblib.dump(modelo, ruta_modelo)

    matriz_df = pd.DataFrame(
        matriz,
        index=[f"Real_{nivel}" for nivel in etiquetas_presentes],
        columns=[f"Predicho_{nivel}" for nivel in etiquetas_presentes],
    )
    matriz_df.to_csv(
        CARPETA_RESULTADOS / "matriz_confusion_ia.csv",
        encoding="utf-8-sig",
    )

    reporte_df = pd.DataFrame(reporte).transpose()
    reporte_df.to_csv(
        CARPETA_RESULTADOS / "reporte_clasificacion_ia.csv",
        encoding="utf-8-sig",
    )

    importancia_df = pd.DataFrame({
        "variable": VARIABLES_IA,
        "importancia": modelo.feature_importances_,
    }).sort_values("importancia", ascending=False)

    importancia_df.to_csv(
        CARPETA_RESULTADOS / "importancia_variables_ia.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metricas = {
        "modelo": "RandomForestClassifier",
        "registros_totales": int(len(df)),
        "registros_entrenamiento": int(len(X_train)),
        "registros_prueba": int(len(X_test)),
        "accuracy": float(accuracy),
        "random_state": 42,
        "variables": VARIABLES_IA,
        "objetivo": OBJETIVO,
        "distribucion_dataset": {
            str(k): int(v) for k, v in conteo.items()
        },
        "nota": (
            "Modelo experimental entrenado para imitar las etiquetas del "
            "motor matemático con datos sintéticos. No es una herramienta "
            "clínica validada."
        ),
    }

    with open(
        CARPETA_RESULTADOS / "metricas_entrenamiento_ia.json",
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(metricas, archivo, indent=4, ensure_ascii=False)

    print("=" * 65)
    print("ENTRENAMIENTO DE RANDOM FOREST FINALIZADO")
    print("=" * 65)
    print(f"Registros totales:       {len(df)}")
    print(f"Registros entrenamiento: {len(X_train)}")
    print(f"Registros prueba:        {len(X_test)}")
    print(f"Accuracy:                {accuracy:.4f} ({accuracy * 100:.2f} %)")
    print("\nReporte de clasificación:")
    print(classification_report(y_test, predicciones, zero_division=0))
    print("Matriz de confusión:")
    print(matriz_df)
    print("\nImportancia de variables:")
    print(importancia_df.to_string(index=False))
    print("\nArchivos guardados en:", CARPETA_RESULTADOS)
    print("Modelo:", ruta_modelo)
    print("\nRecuerda: esta evaluación mide concordancia con el motor matemático,")
    print("no exactitud clínica.")


if __name__ == "__main__":
    entrenar_modelo()
