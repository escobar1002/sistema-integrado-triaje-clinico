# Sistema Integrado de Soporte para Triaje Clínico (V7.8)

Prototipo académico de bajo nivel desarrollado con Python y Gradio. Integra registro de datos ficticios, normalización de variables, un modelo matemático ponderado, un clasificador Random Forest independiente, reglas experimentales de alertas, historial e indicadores.

> **Aviso importante:** este proyecto es exclusivamente académico y demostrativo. Utiliza datos sintéticos y umbrales experimentales. No constituye un dispositivo médico, no tiene validación clínica y no debe utilizarse para diagnóstico, priorización real ni decisiones asistenciales.

## Tecnologías

- Python
- Gradio
- Pandas y NumPy
- scikit-learn y Joblib
- Plotly
- CSV

## Estructura

- `app.py`: interfaz principal.
- `modules/`: validación, normalización, presión arterial, alertas y motores de evaluación/IA.
- `models/` (si aplica): artefactos de modelos; el modelo Random Forest de inferencia se encuentra en `results/`.
- `results/`: modelo entrenado y métricas experimentales.
- `data/dataset_entrenamiento_ia.csv`: conjunto sintético de entrenamiento.
- `data/registros_simulados.csv`: archivo vacío con encabezados para iniciar un historial local.
- `tests/`: pruebas automatizadas.

## Requisitos

Se recomienda Python 3.14, sujeto a compatibilidad de las dependencias fijadas en `requirements.txt`.

## Instalación en Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Después, abre la dirección local que indique Gradio en la terminal (habitualmente `http://127.0.0.1:7860`).

## Pruebas

```powershell
python -m unittest discover -s tests -v
```

## Datos y privacidad

El repositorio no incluye el historial local de pruebas ni conjuntos clínicos de terceros. El archivo `data/registros_simulados.csv` se entrega vacío para que la aplicación genere registros localmente. El dataset de entrenamiento incluido es sintético. No cargues datos personales, clínicos reales, credenciales ni archivos sujetos a restricciones de licencia.

## Resultados experimentales

Las métricas reportadas corresponden a la reproducción de etiquetas generadas por el modelo matemático del proyecto a partir de datos sintéticos. No son medidas de precisión clínica ni evidencia de desempeño en pacientes reales.

## Autoría y contexto

Proyecto académico desarrollado en el marco de Proyecto de Ingeniería I, UNAD.

## Licencia

No se ha asignado una licencia de código abierto. Todos los derechos reservados por sus autores hasta que el equipo defina y agregue una licencia. La visibilidad pública del repositorio no concede por sí sola permiso de reutilización.
