"""
Genera un conjunto de datos sintéticos etiquetado por el motor matemático
actual del Sistema Integrado de Soporte para Triaje Clínico.

IMPORTANTE:
- Los registros son artificiales; no representan pacientes reales.
- Las etiquetas reproducen el índice experimental existente.
- No se modifican app.py, el historial ni los indicadores.
- Los rangos corresponden a límites técnicos de diseño del prototipo,
  no a rangos clínicos de normalidad.
"""

import random
from pathlib import Path

import pandas as pd

from modules.normalizacion import normalizar_paciente
from modules.modelo_triaje import evaluar_modelo_triaje


SEMILLA = 42
CANTIDAD_REGISTROS = 10_000

CARPETA_DATOS = Path(__file__).resolve().parent / "data"
ARCHIVO_SALIDA = CARPETA_DATOS / "dataset_entrenamiento_ia.csv"


def generar_paciente_sintetico(rng):
    """Genera una combinación dentro de los límites técnicos del prototipo."""

    edad = rng.randint(0, 120)
    dolor = rng.randint(0, 10)
    frecuencia_cardiaca = rng.randint(30, 220)
    frecuencia_respiratoria = rng.randint(5, 50)
    temperatura = round(rng.uniform(30, 42), 1)
    spo2 = round(rng.uniform(85, 100), 1)

    # Se garantiza PAS > PAD para cumplir la validación existente.
    presion_diastolica = rng.randint(1, 199)
    presion_sistolica = rng.randint(presion_diastolica + 1, 300)

    return {
        "edad": edad,
        "dolor": dolor,
        "frecuencia_cardiaca": frecuencia_cardiaca,
        "frecuencia_respiratoria": frecuencia_respiratoria,
        "temperatura": temperatura,
        "spo2": spo2,
        "presion_sistolica": presion_sistolica,
        "presion_diastolica": presion_diastolica,
    }


def construir_variables_modelo(normalizados):
    """
    Mantiene exactamente la correspondencia utilizada por app.py
    para alimentar el modelo matemático.
    """
    return {
        "spo2": normalizados["spo2_riesgo"],
        "pa": normalizados["pam_riesgo"],
        "frecuencia_cardiaca": normalizados["frecuencia_cardiaca_riesgo"],
        "frecuencia_respiratoria": normalizados["frecuencia_respiratoria_riesgo"],
        "temperatura": normalizados["temperatura_riesgo"],
        "dolor": normalizados["dolor"],
        "edad": normalizados["edad_riesgo"],
    }


def generar_dataset(cantidad=CANTIDAD_REGISTROS, semilla=SEMILLA):
    rng = random.Random(semilla)
    registros = []

    for numero in range(1, cantidad + 1):
        paciente = generar_paciente_sintetico(rng)

        # Se reutilizan las funciones reales del proyecto.
        normalizados = normalizar_paciente(paciente)
        variables_modelo = construir_variables_modelo(normalizados)
        resultado = evaluar_modelo_triaje(variables_modelo)

        registro = {
            "registro_id": numero,
            **paciente,
            **{
                f"entrada_{nombre}": valor
                for nombre, valor in variables_modelo.items()
            },
            "indice_prioridad": resultado["indice_prioridad"],
            "nivel_experimental": resultado["nivel_experimental"],
        }
        registros.append(registro)

    return pd.DataFrame(registros)


def main():
    CARPETA_DATOS.mkdir(parents=True, exist_ok=True)

    dataset = generar_dataset()
    dataset.to_csv(ARCHIVO_SALIDA, index=False, encoding="utf-8-sig")

    print("=" * 65)
    print("GENERACIÓN DEL DATASET PARA ENTRENAMIENTO DE IA")
    print("=" * 65)
    print(f"Registros generados: {len(dataset)}")
    print(f"Archivo: {ARCHIVO_SALIDA}")
    print("\nDistribución de niveles experimentales:")
    print(dataset["nivel_experimental"].value_counts().sort_index())
    print("\nPrimeros cinco registros:")
    print(dataset.head().to_string(index=False))
    print("\nDataset generado correctamente.")
    print("Estos datos son sintéticos y reproducen el motor matemático.")


if __name__ == "__main__":
    main()
