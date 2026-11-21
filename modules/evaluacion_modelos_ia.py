"""Compara Random Forest y regresion lineal frente al modelo matematico original.
Las etiquetas proceden de datos sinteticos; las metricas NO son validacion clinica.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from modules.modelo_triaje import clasificar_indice_prioridad

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "dataset_entrenamiento_ia.csv"
RF_PATH = BASE_DIR / "results" / "modelo_ia_random_forest.joblib"
RESULTS_DIR = BASE_DIR / "results"

VARIABLES_IA = [
    "entrada_spo2",
    "entrada_pa",
    "entrada_frecuencia_cardiaca",
    "entrada_frecuencia_respiratoria",
    "entrada_temperatura",
    "entrada_dolor",
    "entrada_edad",
]
NIVELES = [
    "Nivel I",
    "Nivel II",
    "Nivel III",
    "Nivel IV",
    "Nivel V"
]


def convertir_indice_a_nivel(indice):
    resultado = clasificar_indice_prioridad(float(indice))

    if isinstance(resultado, dict):
        resultado = resultado.get(
            "nivel_experimental",
            resultado.get("nivel", resultado)
        )
    elif isinstance(resultado, (tuple, list)):
        resultado = resultado[0]

    resultado = str(resultado).strip()

    if not resultado.startswith("Nivel "):
        resultado = f"Nivel {resultado}"

    return resultado


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"No se encontro el dataset: {DATASET_PATH}")
    if not RF_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro {RF_PATH}. Ejecuta primero: python -m modules.entrenamiento_ia"
        )

    datos = pd.read_csv(DATASET_PATH)
    requeridas = VARIABLES_IA + ["nivel_experimental", "indice_prioridad"]
    faltantes = [c for c in requeridas if c not in datos.columns]
    if faltantes:
        raise ValueError("Faltan columnas en el dataset: " + ", ".join(faltantes))
    datos = datos.dropna(subset=requeridas)

    X = datos[VARIABLES_IA]
    y_nivel = datos["nivel_experimental"].astype(str)
    y_indice = datos["indice_prioridad"].astype(float)

    X_train, X_test, y_train, y_test, indice_train, indice_test = train_test_split(
        X, y_nivel, y_indice, test_size=0.20, random_state=42, stratify=y_nivel
    )

    # Random Forest ya entrenado con la misma particion y semilla.
    modelo_rf = joblib.load(RF_PATH)
    pred_rf = modelo_rf.predict(X_test).astype(str)

    # Regresion lineal: estima el indice y lo convierte con el clasificador original.
    modelo_reg = LinearRegression()
    modelo_reg.fit(X_train, indice_train)
    pred_indice = modelo_reg.predict(X_test)
    pred_reg = np.array([convertir_indice_a_nivel(x) for x in pred_indice])

    acc_rf = accuracy_score(y_test, pred_rf)
    acc_reg = accuracy_score(y_test, pred_reg)
    rep_rf = classification_report(y_test, pred_rf, labels=NIVELES, output_dict=True, zero_division=0)
    rep_reg = classification_report(y_test, pred_reg, labels=NIVELES, output_dict=True, zero_division=0)

    for nombre, pred in [("random_forest", pred_rf), ("regresion", pred_reg)]:
        matriz = confusion_matrix(y_test, pred, labels=NIVELES)
        pd.DataFrame(
            matriz,
            index=[f"Real_{n}" for n in NIVELES],
            columns=[f"Predicho_{n}" for n in NIVELES],
        ).to_csv(RESULTS_DIR / f"matriz_confusion_{nombre}.csv", encoding="utf-8-sig")

    comparacion = pd.DataFrame([
        {
            "modelo": "Random Forest",
            "exactitud": acc_rf,
            "precision_macro": rep_rf["macro avg"]["precision"],
            "recall_macro": rep_rf["macro avg"]["recall"],
            "f1_macro": rep_rf["macro avg"]["f1-score"],
            "mae_indice": np.nan,
            "rmse_indice": np.nan,
            "r2_indice": np.nan,
        },
        {
            "modelo": "Regresion lineal + clasificador original",
            "exactitud": acc_reg,
            "precision_macro": rep_reg["macro avg"]["precision"],
            "recall_macro": rep_reg["macro avg"]["recall"],
            "f1_macro": rep_reg["macro avg"]["f1-score"],
            "mae_indice": mean_absolute_error(indice_test, pred_indice),
            "rmse_indice": mean_squared_error(indice_test, pred_indice) ** 0.5,
            "r2_indice": r2_score(indice_test, pred_indice),
        },
    ])
    comparacion.to_csv(RESULTS_DIR / "comparacion_modelos_ia.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rep_rf).transpose().to_csv(RESULTS_DIR / "reporte_random_forest.csv", encoding="utf-8-sig")
    pd.DataFrame(rep_reg).transpose().to_csv(RESULTS_DIR / "reporte_regresion.csv", encoding="utf-8-sig")
    joblib.dump(modelo_reg, RESULTS_DIR / "modelo_ia_regresion_indice.joblib")

    metricas = {
        "registros_dataset": int(len(datos)),
        "registros_prueba": int(len(X_test)),
        "exactitud_random_forest": float(acc_rf),
        "exactitud_regresion_clasificada": float(acc_reg),
        "mae_regresion_indice": float(mean_absolute_error(indice_test, pred_indice)),
        "rmse_regresion_indice": float(mean_squared_error(indice_test, pred_indice) ** 0.5),
        "r2_regresion_indice": float(r2_score(indice_test, pred_indice)),
        "nota": "Datos sinteticos etiquetados por el modelo matematico; no es validacion clinica.",
    }
    with open(RESULTS_DIR / "metricas_comparacion_ia.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=4, ensure_ascii=False)

    print("\n===== EVALUACION COMPARATIVA =====")
    print(f"Registros evaluados: {len(X_test)}")
    print(f"Exactitud Random Forest: {acc_rf:.2%}")
    print(f"Exactitud Regresion + clasificador: {acc_reg:.2%}")
    print(f"MAE del indice: {metricas['mae_regresion_indice']:.4f}")
    print(f"RMSE del indice: {metricas['rmse_regresion_indice']:.4f}")
    print(f"R2 del indice: {metricas['r2_regresion_indice']:.4f}")
    print(f"Resultados guardados en: {RESULTS_DIR}")
    print("Estas metricas no representan precision ni validacion clinica.")


if __name__ == "__main__":
    main()
