"""
Motor de inferencia Random Forest del Sistema Integrado de Soporte
para Triaje Clínico.

Este módulo recibe las mismas siete variables normalizadas que utiliza
el modelo matemático original. No modifica ni reemplaza dicho modelo.

Las probabilidades devueltas son probabilidades internas del clasificador
sobre sus clases aprendidas; no son probabilidades clínicas.
"""

from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_MODELO = BASE_DIR / "results" / "modelo_ia_random_forest.joblib"

VARIABLES_IA = [
    "entrada_spo2",
    "entrada_pa",
    "entrada_frecuencia_cardiaca",
    "entrada_frecuencia_respiratoria",
    "entrada_temperatura",
    "entrada_dolor",
    "entrada_edad",
]

_modelo = None


def cargar_modelo():
    """Carga el modelo una sola vez durante la ejecución de la aplicación."""
    global _modelo

    if _modelo is None:
        if not RUTA_MODELO.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo Random Forest: {RUTA_MODELO}"
            )
        _modelo = joblib.load(RUTA_MODELO)

    return _modelo


def construir_entrada_ia(variables_modelo):
    """
    Adapta el diccionario común del proyecto al esquema exacto
    utilizado durante el entrenamiento de Random Forest.
    """
    correspondencia = {
        "entrada_spo2": "spo2",
        "entrada_pa": "pa",
        "entrada_frecuencia_cardiaca": "frecuencia_cardiaca",
        "entrada_frecuencia_respiratoria": "frecuencia_respiratoria",
        "entrada_temperatura": "temperatura",
        "entrada_dolor": "dolor",
        "entrada_edad": "edad",
    }

    faltantes = [
        origen for origen in correspondencia.values()
        if origen not in variables_modelo
    ]
    if faltantes:
        raise ValueError(
            "Faltan variables normalizadas para Random Forest: "
            + ", ".join(faltantes)
        )

    fila = {
        destino: float(variables_modelo[origen])
        for destino, origen in correspondencia.items()
    }

    for nombre, valor in fila.items():
        if not 0 <= valor <= 1:
            raise ValueError(
                f"La variable {nombre} debe estar entre 0 y 1."
            )

    return pd.DataFrame([fila], columns=VARIABLES_IA)


def predecir_nivel_ia(variables_modelo):
    """
    Predice el nivel experimental I-V y devuelve las probabilidades
    internas por clase del Random Forest.
    """
    modelo = cargar_modelo()
    entrada = construir_entrada_ia(variables_modelo)

    prediccion = str(modelo.predict(entrada)[0])

    probabilidades = {}
    if hasattr(modelo, "predict_proba"):
        valores = modelo.predict_proba(entrada)[0]
        probabilidades = {
            str(clase): float(probabilidad)
            for clase, probabilidad in zip(modelo.classes_, valores)
        }

    return {
        "nivel_predicho": prediccion,
        "probabilidades": probabilidades,
    }
