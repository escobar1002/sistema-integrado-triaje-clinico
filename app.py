
import gradio as gr
from html import escape as html_escape
import pandas as pd
import plotly.express as px

from pathlib import Path
from datetime import datetime
import re

from modules.procesamiento import validar_paciente
from modules.normalizacion import normalizar_paciente
from modules.alertas import detectar_condiciones_criticas
from modules.modelo_triaje import evaluar_modelo_triaje
from modules.modelo_ia import predecir_nivel_ia


# =====================================================
# 1. CONFIGURACIÓN GENERAL
# =====================================================

TITULO = "Sistema Integrado de Soporte para Triaje Clínico"

CARPETA_DATOS = Path(__file__).parent / "data"
ARCHIVO_HISTORIAL = CARPETA_DATOS / "registros_simulados.csv"

CARPETA_DATOS.mkdir(exist_ok=True)


# =====================================================
# 2. FUNCIÓN PARA GUARDAR REGISTROS
# =====================================================

CAMPOS_HISTORIAL = [
    "fecha_hora", "codigo", "tipo_documento", "documento_ficticio",
    "nombres", "apellidos", "edad", "sexo_registrado", "motivo_consulta",
    "sintomas", "observaciones", "dolor", "frecuencia_cardiaca", "spo2",
    "presion_sistolica", "presion_diastolica", "temperatura",
    "frecuencia_respiratoria", "indice_prioridad", "nivel_experimental",
    "cantidad_alertas"
]


def generar_siguiente_codigo():
    """Genera el siguiente código interno SIM-### a partir del historial guardado."""
    prefijo = "SIM-"
    mayor_consecutivo = 0

    if ARCHIVO_HISTORIAL.exists():
        try:
            existente = pd.read_csv(
                ARCHIVO_HISTORIAL,
                dtype={"codigo": "string"},
                keep_default_na=False,
                encoding="utf-8-sig"
            )
            if "codigo" in existente.columns:
                for valor in existente["codigo"].astype(str):
                    coincidencia = re.fullmatch(r"SIM-(\d+)", valor.strip(), flags=re.IGNORECASE)
                    if coincidencia:
                        mayor_consecutivo = max(
                            mayor_consecutivo, int(coincidencia.group(1))
                        )
        except (pd.errors.EmptyDataError, UnicodeDecodeError, pd.errors.ParserError):
            pass

    return f"{prefijo}{mayor_consecutivo + 1:03d}"


def codigo_ya_registrado(codigo):
    """Consulta el CSV directamente para impedir códigos duplicados."""
    if codigo is None or not str(codigo).strip() or not ARCHIVO_HISTORIAL.exists():
        return False
    try:
        existente = pd.read_csv(
            ARCHIVO_HISTORIAL,
            dtype={"codigo": "string"},
            keep_default_na=False,
            encoding="utf-8-sig"
        )
    except pd.errors.EmptyDataError:
        return False

    if "codigo" not in existente.columns:
        return False

    objetivo = str(codigo).strip().casefold()
    codigos = existente["codigo"].astype(str).str.strip().str.casefold()
    return bool((codigos == objetivo).any())


def guardar_historial(datos):
    """Guarda un registro solo si su código aún no existe en el CSV."""
    codigo = datos.get("codigo", "")
    if codigo_ya_registrado(codigo):
        return False

    nuevo = pd.DataFrame([datos])
    if ARCHIVO_HISTORIAL.exists():
        try:
            existente = pd.read_csv(ARCHIVO_HISTORIAL, encoding="utf-8-sig")
        except (pd.errors.EmptyDataError, UnicodeDecodeError, pd.errors.ParserError):
            existente = pd.DataFrame()
        combinado = pd.concat([existente, nuevo], ignore_index=True, sort=False)
    else:
        combinado = nuevo

    columnas = [c for c in CAMPOS_HISTORIAL if c in combinado.columns]
    columnas += [c for c in combinado.columns if c not in columnas]
    combinado = combinado.reindex(columns=columnas)
    combinado.to_csv(ARCHIVO_HISTORIAL, index=False, encoding="utf-8-sig")
    return True


def cargar_historial():
    if ARCHIVO_HISTORIAL.exists():
        try:
            historial = pd.read_csv(ARCHIVO_HISTORIAL)
        except pd.errors.EmptyDataError:
            historial = pd.DataFrame()
    else:
        historial = pd.DataFrame()

    # Los registros antiguos siguen siendo válidos; sus nuevos campos quedan vacíos.
    for campo in CAMPOS_HISTORIAL:
        if campo not in historial.columns:
            historial[campo] = ""
    return historial



def cargar_historial_vista():
    """Tabla resumida para consulta rápida; el CSV mantiene todos los campos."""
    historial = cargar_historial()
    columnas = [
        "Fecha y hora", "Código", "Paciente", "Identificación", "Edad",
        "Motivo de consulta", "Nivel matemático", "Nivel IA", "Alertas"
    ]
    if historial.empty:
        return pd.DataFrame(columns=columnas)

    vista = historial.copy()
    vista["Paciente"] = (
        vista["nombres"].fillna("").astype(str).str.strip() + " " +
        vista["apellidos"].fillna("").astype(str).str.strip()
    ).str.strip()
    vista["Identificación"] = (
        vista["tipo_documento"].fillna("").astype(str).str.strip() + " " +
        vista["documento_ficticio"].fillna("").astype(str).str.strip()
    ).str.strip()
    vista["Fecha y hora"] = vista["fecha_hora"].fillna("")
    vista["Código"] = vista["codigo"].fillna("")
    vista["Edad"] = vista["edad"].astype("object").fillna("")
    vista["Motivo de consulta"] = vista["motivo_consulta"].fillna("")
    vista["Nivel matemático"] = vista["nivel_experimental"].fillna("")
    vista["Alertas"] = vista["cantidad_alertas"].astype("object").fillna("")

    niveles_ia = []
    for _, fila in vista.iterrows():
        try:
            datos = {
                "edad": float(fila["edad"]),
                "dolor": float(fila["dolor"]),
                "frecuencia_cardiaca": float(fila["frecuencia_cardiaca"]),
                "spo2": float(fila["spo2"]),
                "presion_sistolica": float(fila["presion_sistolica"]),
                "presion_diastolica": float(fila["presion_diastolica"]),
                "temperatura": float(fila["temperatura"]),
                "frecuencia_respiratoria": float(fila["frecuencia_respiratoria"])
            }
            normalizados = normalizar_paciente(datos)
            variables = construir_variables_modelo(normalizados)
            niveles_ia.append(predecir_nivel_ia(variables)["nivel_predicho"])
        except Exception:
            niveles_ia.append("No disponible")
    vista["Nivel IA"] = niveles_ia

    return vista[columnas].fillna("")


def consultar_ficha_historial(codigo):
    """Construye una ficha de consulta con datos guardados y resultados reproducidos."""
    if not codigo or not str(codigo).strip():
        return '<div class="clinical-empty">Ingresa el código del paciente que deseas consultar.</div>'

    historial = cargar_historial()
    if historial.empty:
        return '<div class="clinical-empty">Todavía no existen registros almacenados.</div>'

    coincidencias = historial[
        historial["codigo"].astype(str).str.strip() == str(codigo).strip()
    ]
    if coincidencias.empty:
        return f'<div class="clinical-empty">No se encontró el registro {html_escape(str(codigo))}.</div>'

    datos = coincidencias.iloc[-1].to_dict()
    def valor(campo, predeterminado="No registrado"):
        dato = datos.get(campo, "")
        if pd.isna(dato) or str(dato).strip() in ("", "nan", "None"):
            return predeterminado
        return str(dato).strip()

    def ficha_card(etiqueta, contenido):
        return (
            '<div class="clinical-vital">'
            f'<div class="clinical-vital-label">{html_escape(etiqueta)}</div>'
            f'<div class="clinical-vital-value">{html_escape(str(contenido))}</div>'
            '</div>'
        )

    try:
        datos_modelo = {
            "edad": float(datos["edad"]),
            "dolor": float(datos["dolor"]),
            "frecuencia_cardiaca": float(datos["frecuencia_cardiaca"]),
            "spo2": float(datos["spo2"]),
            "presion_sistolica": float(datos["presion_sistolica"]),
            "presion_diastolica": float(datos["presion_diastolica"]),
            "temperatura": float(datos["temperatura"]),
            "frecuencia_respiratoria": float(datos["frecuencia_respiratoria"])
        }
        normalizados = normalizar_paciente(datos_modelo)
        variables = construir_variables_modelo(normalizados)
        matematico = evaluar_modelo_triaje(variables)
        try:
            nivel_ia = predecir_nivel_ia(variables)["nivel_predicho"]
            comparacion = "Coincidencia" if nivel_ia == matematico["nivel_experimental"] else "Discrepancia"
        except Exception:
            nivel_ia = "No disponible"
            comparacion = "No fue posible comparar los motores"
        resultado_alertas = detectar_condiciones_criticas(datos_modelo)
        indice = f'{matematico["indice_prioridad"]:.4f}'
        nivel_matematico = matematico["nivel_experimental"]
        tarjetas_signos = [
            ficha_card("Edad", f'{datos_modelo["edad"]:g} años'),
            ficha_card("Dolor", f'{datos_modelo["dolor"]:g}/10'),
            ficha_card("Frecuencia cardíaca", f'{datos_modelo["frecuencia_cardiaca"]:g} bpm'),
            ficha_card("Saturación SpO₂", f'{datos_modelo["spo2"]:g} %'),
            ficha_card("Presión arterial", f'{datos_modelo["presion_sistolica"]:g}/{datos_modelo["presion_diastolica"]:g} mmHg'),
            ficha_card("Temperatura", f'{datos_modelo["temperatura"]:g} °C'),
            ficha_card("Frecuencia respiratoria", f'{datos_modelo["frecuencia_respiratoria"]:g} rpm')
        ]
        pam = resultado_alertas.get("evaluacion_pam", {}).get("pam")
        if pam is not None:
            tarjetas_signos.append(ficha_card("PAM calculada", f"{pam} mmHg"))
        alertas_html = "".join(
            '<div class="clinical-alert-item"><strong>'
            + html_escape(str(a.get("variable", "Alerta"))) + '</strong> · '
            + html_escape(str(a.get("valor", ""))) + '<br>'
            + html_escape(str(a.get("mensaje", ""))) + '</div>'
            for a in resultado_alertas.get("alertas", [])
        )
        if not alertas_html:
            alertas_html = '<div class="clinical-alert-item">Sin alertas según las reglas experimentales implementadas.</div>'
    except Exception as error:
        indice = "No disponible"
        nivel_matematico = valor("nivel_experimental")
        nivel_ia = "No disponible"
        comparacion = "No fue posible recalcular los motores"
        tarjetas_signos = [
            ficha_card("Edad", valor("edad")),
            ficha_card("Dolor", valor("dolor")),
            ficha_card("Frecuencia cardíaca", valor("frecuencia_cardiaca")),
            ficha_card("Saturación SpO₂", valor("spo2")),
            ficha_card("Presión arterial", f'{valor("presion_sistolica")}/{valor("presion_diastolica")}'),
            ficha_card("Temperatura", valor("temperatura")),
            ficha_card("Frecuencia respiratoria", valor("frecuencia_respiratoria"))
        ]
        alertas_html = (
            '<div class="clinical-alert-item">No fue posible recalcular los resultados: '
            + html_escape(str(error)) + '</div>'
        )

    nombre = (valor("nombres", "") + " " + valor("apellidos", "")).strip() or "No registrado"
    identificacion = (valor("tipo_documento", "") + " " + valor("documento_ficticio", "")).strip() or "No registrada"
    signos_html = "".join(tarjetas_signos)

    return f"""
    <div class="clinical-results">
      <div class="clinical-result-heading">
        <div>
          <div class="clinical-eyebrow">FICHA INDIVIDUAL · REGISTRO SIMULADO</div>
          <h2>{html_escape(nombre)}</h2>
          <p>{html_escape(valor("codigo"))} · {html_escape(valor("fecha_hora"))}</p>
        </div>
        <div class="clinical-alert-count">{html_escape(valor("cantidad_alertas", "0"))} alerta(s) registradas</div>
      </div>
      <h3>Identificación e ingreso</h3>
      <div class="clinical-vitals">
        {ficha_card("Identificación ficticia", identificacion)}
        {ficha_card("Edad", valor("edad") + " años")}
        {ficha_card("Sexo registrado", valor("sexo_registrado"))}
      </div>
      <div class="clinical-alert-item">
        <strong>Motivo de consulta:</strong> {html_escape(valor("motivo_consulta"))}<br>
        <strong>Síntomas:</strong> {html_escape(valor("sintomas", "No especificados"))}<br>
        <strong>Observaciones:</strong> {html_escape(valor("observaciones", "Sin observaciones"))}
      </div>
      <h3>Signos y variables registradas</h3>
      <div class="clinical-vitals">{signos_html}</div>
      <h3>Resultados de los motores</h3>
      <div class="clinical-model-grid">
        <div class="clinical-model-card"><span>MODELO MATEMÁTICO</span><strong>{html_escape(str(nivel_matematico))}</strong><small>Índice experimental: {html_escape(str(indice))}</small></div>
        <div class="clinical-model-card"><span>RANDOM FOREST</span><strong>{html_escape(str(nivel_ia))}</strong><small>Predicción independiente</small></div>
      </div>
      <div class="clinical-comparison {'match' if comparacion == 'Coincidencia' else 'mismatch'}">
        <strong>{html_escape(comparacion)}</strong>
        <span>Resultados experimentales; no determinan prioridad clínica real.</span>
      </div>
      <h3>Alertas registradas</h3>
      {alertas_html}
      <p class="clinical-disclaimer">Prototipo académico con datos ficticios. No está validado clínicamente y no debe utilizarse para decisiones asistenciales.</p>
    </div>
    """

# =====================================================
# 3.1 EVALUACIÓN DE PACIENTES
# =====================================================


def construir_variables_modelo(normalizados):
    """
    Prepara las siete variables de entrada del modelo matemático.
    Todos los valores se orientan hacia el riesgo.
    """
    campos_requeridos = [
        "spo2_riesgo",
        "pam_riesgo",
        "frecuencia_cardiaca_riesgo",
        "frecuencia_respiratoria_riesgo",
        "temperatura_riesgo",
        "dolor",
        "edad_riesgo"
    ]

    faltantes = [campo for campo in campos_requeridos if campo not in normalizados]
    if faltantes:
        raise ValueError(
            "No se puede calcular el índice. Faltan valores: "
            + ", ".join(faltantes)
        )

    return {
        "spo2": normalizados["spo2_riesgo"],
        "pa": normalizados["pam_riesgo"],
        "frecuencia_cardiaca": normalizados["frecuencia_cardiaca_riesgo"],
        "frecuencia_respiratoria": normalizados["frecuencia_respiratoria_riesgo"],
        "temperatura": normalizados["temperatura_riesgo"],
        "dolor": normalizados["dolor"],
        "edad": normalizados["edad_riesgo"]
    }


def evaluar_paciente_detallado(codigo):

    if not codigo or not codigo.strip():
        return "Debes ingresar un código de paciente."

    historial = cargar_historial()

    if historial.empty:
        return "Todavía no existen registros para evaluar."

    paciente = historial[
        historial["codigo"].astype(str).str.strip() == codigo.strip()
    ]

    if paciente.empty:
        return f"No se encontró el paciente {codigo}."

    datos = paciente.iloc[-1].to_dict()

    campos = [
        "edad", "dolor", "frecuencia_cardiaca", "spo2",
        "presion_sistolica", "presion_diastolica",
        "temperatura", "frecuencia_respiratoria"
    ]

    for campo in campos:
        datos[campo] = float(datos[campo])

    normalizados = normalizar_paciente(datos)
    variables_modelo = construir_variables_modelo(normalizados)
    resultado_modelo = evaluar_modelo_triaje(variables_modelo)

    # Segundo motor independiente: Random Forest.
    # Recibe exactamente las mismas variables normalizadas del modelo matemático.
    try:
        resultado_ia = predecir_nivel_ia(variables_modelo)
        nivel_ia = resultado_ia["nivel_predicho"]
        coincide_modelos = (
            nivel_ia == resultado_modelo["nivel_experimental"]
        )
        estado_comparacion = (
            "COINCIDENCIA" if coincide_modelos else "DISCREPANCIA"
        )
        probabilidades_ia = resultado_ia["probabilidades"]

        if probabilidades_ia:
            filas_probabilidades = "\n".join(
                f"| {nivel} | {probabilidad:.2%} |"
                for nivel, probabilidad in sorted(
                    probabilidades_ia.items(),
                    key=lambda elemento: elemento[0]
                )
            )
            tabla_probabilidades_ia = (
                "\n| Clase predicha por el modelo | Probabilidad interna |\n"
                "|---|---:|\n"
                + filas_probabilidades
            )
        else:
            tabla_probabilidades_ia = (
                "El clasificador no expuso probabilidades por clase."
            )

        seccion_ia = f"""
### 2. Predicción del motor de inteligencia artificial (Random Forest)

| Motor | Resultado |
|---|---|
| Modelo matemático original | **{resultado_modelo['nivel_experimental']}** |
| Random Forest | **{nivel_ia}** |
| Comparación | **{estado_comparacion}** |

{tabla_probabilidades_ia}

<small>Las probabilidades son estimaciones internas del clasificador sobre las clases sintéticas aprendidas; no representan probabilidades clínicas.</small>
"""
    except Exception as error_ia:
        seccion_ia = f"""
### 2. Predicción del motor de inteligencia artificial (Random Forest)

No fue posible ejecutar el modelo de IA: `{type(error_ia).__name__}: {error_ia}`

El resultado del modelo matemático original se conserva sin cambios.
"""

    resultado_alertas = detectar_condiciones_criticas(datos)

    filas = ""
    for variable, valor in normalizados.items():
        filas += f"| {variable} | {valor} |\n"

    if resultado_alertas["alertas"]:
        texto_alertas = ""
        for alerta in resultado_alertas["alertas"]:
            indicador = "🔴" if alerta["nivel"] == "ROJO" else "🟡"
            texto_alertas += (
                f"- {indicador} **{alerta['variable']}:** "
                f"{alerta['valor']}. {alerta['mensaje']}\n"
            )
    else:
        texto_alertas = (
            "🟢 No se detectaron alertas según "
            "los criterios implementados."
        )

    evaluacion_pam = resultado_alertas["evaluacion_pam"]
    color_pam = {
        "ROJO": "🔴",
        "AMARILLO": "🟡",
        "VERDE": "🟢"
    }.get(evaluacion_pam["nivel"], "⚪")

    if evaluacion_pam["pam"] is not None:
        pam_texto = f"""
### Evaluación de presión arterial media (PAM)

| Indicador | Resultado |
|---|---:|
| Presión sistólica | {datos['presion_sistolica']} mmHg |
| Presión diastólica | {datos['presion_diastolica']} mmHg |
| PAM calculada | **{evaluacion_pam['pam']} mmHg** |
| PAM normalizada | {normalizados['pam_normalizada']} |
| PAM orientada al riesgo | {normalizados['pam_riesgo']} |

**Estado:** {color_pam} **{evaluacion_pam['estado']}**

{evaluacion_pam['mensaje']}
"""
    else:
        pam_texto = """
### Evaluación de presión arterial media (PAM)

No fue posible calcular la PAM porque faltan los valores
de presión sistólica o diastólica.
"""

    filas_contribuciones = ""
    for variable, aporte in resultado_modelo["contribuciones"].items():
        coeficiente = resultado_modelo["coeficientes"][variable]
        filas_contribuciones += (
            f"| {variable} | {variables_modelo[variable]:.4f} | "
            f"{coeficiente:.2f} | {aporte:.4f} |\n"
        )

    informe = f"""
## Resultado del procesamiento

**Paciente:** {codigo}

**Fecha del registro:** {datos['fecha_hora']}

### 1. Índice de prioridad experimental

# {resultado_modelo['indice_prioridad']:.4f}

**Clasificación matemática:** {resultado_modelo['nivel_experimental']}

| Variable | Valor de riesgo | Peso | Contribución |
|---|---:|---:|---:|
{filas_contribuciones}

{seccion_ia}

### 3. Normalización de variables

| Variable | Valor |
|---|---:|
{filas}

{pam_texto}

### 4. Detección de alertas

**Estado general:** {resultado_alertas['estado']}

**Cantidad de alertas:** {resultado_alertas['cantidad_alertas']}

{texto_alertas}

### 5. Consideraciones técnicas

- El índice se calcula con los coeficientes establecidos en la Etapa 3.
- Las variables fisiológicas se transforman mediante reglas experimentales orientadas al riesgo.
- La edad aporta al índice mediante una transformación experimental.
- Los niveles I–V corresponden a intervalos matemáticos del prototipo; no equivalen a categorías oficiales de triaje.
- La PAM se calcula a partir de la presión sistólica y diastólica.

---

**Advertencia:** prototipo académico con datos simulados. No constituye diagnóstico,
no está validado clínicamente y no debe utilizarse para decisiones de atención.
"""
    return informe


def evaluar_paciente(codigo):
    """Vista de lectura rápida + informe técnico sin alterar los motores."""
    detalle = evaluar_paciente_detallado(codigo)
    if not codigo or not str(codigo).strip() or detalle.startswith(("Debes", "Todavía", "No se encontró")):
        return f'<div class="clinical-empty">{html_escape(detalle)}</div>', ""
    historial = cargar_historial()
    paciente = historial[historial["codigo"].astype(str).str.strip() == str(codigo).strip()]
    if paciente.empty:
        return '<div class="clinical-empty">Registro no encontrado.</div>', ""
    datos = paciente.iloc[-1].to_dict()
    for campo in ["edad", "dolor", "frecuencia_cardiaca", "spo2", "presion_sistolica", "presion_diastolica", "temperatura", "frecuencia_respiratoria"]:
        datos[campo] = float(datos[campo])
    normalizados = normalizar_paciente(datos)
    variables = construir_variables_modelo(normalizados)
    matematico = evaluar_modelo_triaje(variables)
    alertas = detectar_condiciones_criticas(datos)
    try:
        prediccion = predecir_nivel_ia(variables)["nivel_predicho"]
        comparacion = "Coincidencia" if prediccion == matematico["nivel_experimental"] else "Discrepancia"
        estado_clase = "match" if comparacion == "Coincidencia" else "mismatch"
    except Exception:
        prediccion = "No disponible"
        comparacion = "IA no disponible"
        estado_clase = "mismatch"

    def card(label, value, unit=""):
        return (f'<div class="clinical-vital"><div class="clinical-vital-label">{html_escape(label)}</div>'
                f'<div class="clinical-vital-value">{html_escape(str(value))} <small>{html_escape(unit)}</small></div></div>')

    nivel = html_escape(str(matematico["nivel_experimental"]))
    pred = html_escape(str(prediccion))
    cards = "".join([
        card("Edad", f'{datos["edad"]:g}', "años"),
        card("Dolor registrado", f'{datos["dolor"]:g}/10'),
        card("Frecuencia cardíaca", f'{datos["frecuencia_cardiaca"]:g}', "bpm"),
        card("Saturación SpO₂", f'{datos["spo2"]:g}', "%"),
        card("Presión arterial", f'{datos["presion_sistolica"]:g}/{datos["presion_diastolica"]:g}', "mmHg"),
        card("Temperatura", f'{datos["temperatura"]:g}', "°C"),
        card("Frecuencia respiratoria", f'{datos["frecuencia_respiratoria"]:g}', "rpm"),
    ])
    lista_alertas = []
    for alerta in alertas.get("alertas", []):
        lista_alertas.append(
            '<div class="clinical-alert-item"><strong>' + html_escape(str(alerta.get("variable", "Alerta"))) + '</strong>'
            + ' · ' + html_escape(str(alerta.get("valor", ""))) + '<div>'
            + html_escape(str(alerta.get("mensaje", ""))) + '</div></div>'
        )
    if not lista_alertas:
        lista_alertas = ['<div class="clinical-alert-item">No se detectaron alertas según las reglas experimentales implementadas.</div>']
    alerta_clase = "clinical-alert-active" if alertas.get("cantidad_alertas", 0) else "clinical-alert-clear"
    resumen = f"""<div class="clinical-results">
      <div class="clinical-result-heading"><div><div class="clinical-eyebrow">RESULTADO DEL CASO SIMULADO</div>
      <h2>Paciente {html_escape(str(codigo))}</h2><p>Registro: {html_escape(str(datos.get("fecha_hora", "Sin fecha")))}</p></div>
      <div class="clinical-alert-count">{alertas.get("cantidad_alertas", 0)} alerta(s)</div></div>
      <div class="clinical-model-grid">
        <div class="clinical-model-card"><span>MODELO MATEMÁTICO</span><strong>{nivel}</strong><small>Índice experimental: {matematico["indice_prioridad"]:.4f}</small></div>
        <div class="clinical-model-card"><span>RANDOM FOREST</span><strong>{pred}</strong><small>Clasificación experimental independiente</small></div>
      </div>
      <div class="clinical-comparison {estado_clase}"><strong>{html_escape(comparacion)}</strong><span>Los modelos no sustituyen el criterio profesional ni determinan una prioridad clínica real.</span></div>
      <h3>Identificación e información de ingreso</h3>
      <div class="clinical-vitals">
        {card("Nombre", (str(datos.get("nombres", "")).strip() + " " + str(datos.get("apellidos", "")).strip()).strip() or "No registrado")}
        {card("Documento ficticio", (str(datos.get("tipo_documento", "")).strip() + " " + str(datos.get("documento_ficticio", "")).strip()).strip() or "No registrado")}
        {card("Sexo registrado", str(datos.get("sexo_registrado", "")).strip() or "No registrado")}
      </div>
      <div class="clinical-alert-item"><strong>Motivo de consulta:</strong> {html_escape(str(datos.get("motivo_consulta", "")).strip() or "No registrado")}<br><strong>Síntomas:</strong> {html_escape(str(datos.get("sintomas", "")).strip() or "No especificados")}<br><strong>Observaciones:</strong> {html_escape(str(datos.get("observaciones", "")).strip() or "Sin observaciones")}</div>
      <h3>Signos y datos registrados</h3><div class="clinical-vitals">{cards}</div>
      <div class="clinical-alert-box {alerta_clase}"><h3>Alertas del prototipo</h3>{''.join(lista_alertas)}</div>
      <p class="clinical-disclaimer">Uso académico con datos ficticios. Los umbrales, alertas y niveles I–V no están validados clínicamente.</p>
    </div>"""
    return resumen, detalle


# =====================================================
# 3.2 GENERACIÓN DE INDICADORES
# =====================================================

def generar_indicadores():

    historial = cargar_historial()

    if historial.empty:
        return (
            "## Panel de indicadores\n\nTodavía no existen registros para analizar.",
            None, None, None, None
        )

    total_registros = len(historial)
    total_pacientes = historial["codigo"].nunique()

    registros_con_alertas = 0
    conteo_alertas = {}
    filas_edad = []

    for _, fila in historial.iterrows():
        try:
            datos = {
                "edad": float(fila["edad"]),
                "spo2": float(fila["spo2"]),
                "temperatura": float(fila["temperatura"]),
                "presion_sistolica": float(fila["presion_sistolica"]),
                "presion_diastolica": float(fila["presion_diastolica"])
            }

            resultado = detectar_condiciones_criticas(datos)
            tiene_alertas = resultado["cantidad_alertas"] > 0

            if tiene_alertas:
                registros_con_alertas += 1

            for alerta in resultado["alertas"]:
                variable = alerta["variable"]
                conteo_alertas[variable] = conteo_alertas.get(variable, 0) + 1

            edad = datos["edad"]
            if edad < 18:
                grupo = "0–17"
            elif edad < 40:
                grupo = "18–39"
            elif edad < 60:
                grupo = "40–59"
            elif edad < 80:
                grupo = "60–79"
            else:
                grupo = "80+"

            # Se vuelve a calcular el nivel con los mismos módulos del sistema.
            datos_modelo = {
                **datos,
                "dolor": float(fila["dolor"]),
                "frecuencia_cardiaca": float(fila["frecuencia_cardiaca"]),
                "frecuencia_respiratoria": float(fila["frecuencia_respiratoria"])
            }
            normalizados = normalizar_paciente(datos_modelo)
            variables_modelo = construir_variables_modelo(normalizados)
            modelo = evaluar_modelo_triaje(variables_modelo)

            filas_edad.append({
                "grupo_edad": grupo,
                "alerta": tiene_alertas,
                "nivel": modelo["nivel_experimental"]
            })

        except (ValueError, TypeError, KeyError):
            continue

    historial_fechas = historial.copy()
    historial_fechas["fecha_hora"] = pd.to_datetime(
        historial_fechas["fecha_hora"], errors="coerce"
    )
    historial_fechas = historial_fechas.dropna(subset=["fecha_hora"])
    historial_fechas["fecha"] = historial_fechas["fecha_hora"].dt.strftime("%Y-%m-%d")

    registros_por_fecha = (
        historial_fechas.groupby("fecha").size().reset_index(name="cantidad")
    )

    grafico_fechas = None
    if not registros_por_fecha.empty:
        grafico_fechas = px.bar(
            registros_por_fecha, x="fecha", y="cantidad",
            title="Registros simulados por fecha",
            labels={"fecha": "Fecha", "cantidad": "Cantidad de registros"},
            text="cantidad"
        )

    grafico_alertas = None
    if conteo_alertas:
        datos_alertas = pd.DataFrame(
            list(conteo_alertas.items()),
            columns=["Variable", "Cantidad"]
        )
        grafico_alertas = px.pie(
            datos_alertas, names="Variable", values="Cantidad",
            title="Distribución de alertas detectadas", hole=0.35
        )

    df_edad = pd.DataFrame(filas_edad)
    grafico_alertas_edad = None
    grafico_niveles_edad = None
    resumen_edad = "No hay registros válidos para el análisis por edad."

    if not df_edad.empty:
        orden_grupos = ["0–17", "18–39", "40–59", "60–79", "80+"]
        resumen_grupos = (
            df_edad.groupby("grupo_edad")
            .agg(total=("alerta", "size"), con_alertas=("alerta", "sum"))
            .reindex(orden_grupos, fill_value=0)
            .reset_index()
        )
        resumen_grupos["porcentaje_alertas"] = (
            resumen_grupos.apply(
                lambda r: (r["con_alertas"] / r["total"] * 100)
                if r["total"] else 0,
                axis=1
            )
        )

        grafico_alertas_edad = px.bar(
            resumen_grupos, x="grupo_edad", y="porcentaje_alertas",
            title="Porcentaje de registros con alertas por grupo de edad",
            labels={
                "grupo_edad": "Grupo de edad",
                "porcentaje_alertas": "% de registros con alertas"
            },
            text_auto=".1f"
        )

        distribucion = (
            df_edad.groupby(["grupo_edad", "nivel"]).size()
            .reset_index(name="cantidad")
        )
        grafico_niveles_edad = px.bar(
            distribucion, x="grupo_edad", y="cantidad", color="nivel",
            barmode="stack",
            category_orders={"grupo_edad": orden_grupos},
            title="Distribución de niveles experimentales por edad",
            labels={
                "grupo_edad": "Grupo de edad",
                "cantidad": "Cantidad de registros",
                "nivel": "Nivel experimental"
            }
        )

        resumen_edad = resumen_grupos.to_string(index=False, float_format=lambda x: f"{x:.2f}")

    porcentaje_alertas = (
        registros_con_alertas / total_registros * 100
    ) if total_registros else 0

    resumen = f"""
## Panel de indicadores

### Resumen general

| Indicador | Resultado |
|---|---:|
| Total de registros | {total_registros} |
| Pacientes diferentes | {total_pacientes} |
| Registros con alertas | {registros_con_alertas} |
| Registros sin alertas | {total_registros - registros_con_alertas} |
| Porcentaje de registros con alertas | {porcentaje_alertas:.2f}% |

### Estadísticas por grupo de edad

Los porcentajes se calculan sobre el total de registros válidos de cada grupo,
no mediante comparación de conteos absolutos.

{resumen_edad}

### Interpretación y limitaciones

Los indicadores describen únicamente los registros simulados y las reglas
programadas. Las alertas son generadas por el propio prototipo y no demuestran
una asociación clínica entre edad y riesgo. Los niveles I–V son experimentales,
no categorías clínicas oficiales.
"""
    return (
        resumen, grafico_fechas, grafico_alertas,
        grafico_alertas_edad, grafico_niveles_edad
    )


# =====================================================
# 4. REGISTRO Y VALIDACIÓN DE PACIENTES
# =====================================================

def validar_etapa_identificacion(codigo, tipo_documento, documento, nombres, apellidos, edad, sexo):
    errores = []
    if not codigo or not str(codigo).strip():
        errores.append("Ingresa un código interno, por ejemplo SIM-010.")
    if not tipo_documento:
        errores.append("Selecciona el tipo de documento ficticio.")
    if not documento or not str(documento).strip():
        errores.append("Ingresa un identificador documental ficticio.")
    if not nombres or not str(nombres).strip():
        errores.append("Ingresa los nombres ficticios.")
    if not apellidos or not str(apellidos).strip():
        errores.append("Ingresa los apellidos ficticios.")
    if edad is None:
        errores.append("Ingresa la edad.")
    if not sexo:
        errores.append("Selecciona el sexo registrado.")
    if errores:
        return (
            "### Completa la identificación antes de continuar\\n\\n" +
            "\\n".join(f"- {e}" for e in errores),
            gr.update(visible=True),
            gr.update(visible=False)
        )
    return (
        "### Identificación completa\\nPuedes continuar con el registro de signos y variables.",
        gr.update(visible=False),
        gr.update(visible=True)
    )


def registrar_paciente(
    codigo, tipo_documento, documento, nombres, apellidos, edad, sexo,
    motivo_consulta, sintomas, observaciones,
    dolor, fc, spo2, sistolica, diastolica, temperatura, fr
):
    datos = {
        "edad": edad,
        "dolor": dolor,
        "frecuencia_cardiaca": fc,
        "spo2": spo2,
        "presion_sistolica": sistolica,
        "presion_diastolica": diastolica,
        "temperatura": temperatura,
        "frecuencia_respiratoria": fr
    }

    valido, errores = validar_paciente(datos)
    if not codigo or not str(codigo).strip():
        errores.insert(0, "Debes ingresar un código interno.")
    if not nombres or not str(nombres).strip():
        errores.append("Debes ingresar los nombres ficticios.")
    if not apellidos or not str(apellidos).strip():
        errores.append("Debes ingresar los apellidos ficticios.")
    if not motivo_consulta or not str(motivo_consulta).strip():
        errores.append("Debes ingresar el motivo de consulta.")
    if not tipo_documento or not documento or not str(documento).strip():
        errores.append("Completa el tipo y número de documento ficticio.")
    if not sexo:
        errores.append("Selecciona el sexo registrado.")

    # Evitar duplicados: cada código identifica un único registro en esta versión.
    if codigo_ya_registrado(codigo):
        errores.append(
            f"El código {str(codigo).strip()} ya existe en el historial. "
            "No se guardó ninguna información. Consulta el registro existente "
            "o utiliza un código nuevo."
        )

    if errores:
        mensaje = "### No se pudo guardar el registro\\n\\n" + "\\n".join(f"- {e}" for e in errores) + "\\n\\n**No se agregó ninguna fila al historial.**"
        return (
            mensaje, None,
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(),
            gr.update(visible=True), gr.update(visible=False)
        )

    normalizados = normalizar_paciente(datos)
    variables_modelo = construir_variables_modelo(normalizados)
    resultado_modelo = evaluar_modelo_triaje(variables_modelo)
    resultado_alertas = detectar_condiciones_criticas(datos)

    registro = {
        "fecha_hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "codigo": str(codigo).strip(),
        "tipo_documento": str(tipo_documento).strip(),
        "documento_ficticio": str(documento).strip(),
        "nombres": str(nombres).strip(),
        "apellidos": str(apellidos).strip(),
        "edad": edad,
        "sexo_registrado": str(sexo).strip(),
        "motivo_consulta": str(motivo_consulta).strip(),
        "sintomas": str(sintomas or "").strip(),
        "observaciones": str(observaciones or "").strip(),
        "dolor": dolor,
        "frecuencia_cardiaca": fc,
        "spo2": spo2,
        "presion_sistolica": sistolica,
        "presion_diastolica": diastolica,
        "temperatura": temperatura,
        "frecuencia_respiratoria": fr,
        "indice_prioridad": resultado_modelo["indice_prioridad"],
        "nivel_experimental": resultado_modelo["nivel_experimental"],
        "cantidad_alertas": resultado_alertas["cantidad_alertas"]
    }
    guardado = guardar_historial(registro)
    if not guardado:
        mensaje = (
            f"### No se pudo guardar el registro\\n\\n"
            f"El código **{str(codigo).strip()}** ya existe en el historial. "
            "No se agregó ninguna fila. Utiliza un código nuevo."
        )
        return (
            mensaje, None,
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(),
            gr.update(visible=True), gr.update(visible=False)
        )

    resumen = pd.DataFrame({
        "Dato": [
            "Código interno", "Nombre completo", "Documento ficticio", "Edad",
            "Sexo registrado", "Motivo de consulta", "Síntomas", "Dolor",
            "Frecuencia cardíaca", "SpO₂", "Presión arterial", "Temperatura",
            "Frecuencia respiratoria"
        ],
        "Información": [
            registro["codigo"],
            f'{registro["nombres"]} {registro["apellidos"]}',
            f'{registro["tipo_documento"]}: {registro["documento_ficticio"]}',
            f'{edad} años', registro["sexo_registrado"], registro["motivo_consulta"],
            registro["sintomas"] or "No especificados", f"{dolor}/10",
            f"{fc} bpm", f"{spo2} %", f"{sistolica}/{diastolica} mmHg",
            f"{temperatura} °C", f"{fr} rpm"
        ]
    })
    mensaje = (
        "### Registro guardado correctamente\\n\\n"
        f'**Paciente:** {registro["nombres"]} {registro["apellidos"]} '
        f'({registro["codigo"]})\\n\\n'
        "**Estado:** datos dentro de los límites técnicos de captura.\\n\\n"
        "**Almacenamiento:** registro incorporado al historial local.\\n\\n"
        "Los datos personales son ficticios y la validación informática no representa una valoración clínica."
    )
    # Limpiar exactamente los 17 campos del formulario y volver a la etapa 1.
    return (
        mensaje, resumen,
        gr.update(value=""),                    # codigo
        gr.update(value="Documento ficticio"), # tipo_documento
        gr.update(value=""),                    # documento
        gr.update(value=""),                    # nombres
        gr.update(value=""),                    # apellidos
        gr.update(value=None),                  # edad
        gr.update(value=None),                  # sexo
        gr.update(value=""),                    # motivo_consulta
        gr.update(value=""),                    # sintomas
        gr.update(value=""),                    # observaciones
        gr.update(value=None),                  # dolor
        gr.update(value=None),                  # frecuencia_cardiaca
        gr.update(value=None),                  # spo2
        gr.update(value=None),                  # sistolica
        gr.update(value=None),                  # diastolica
        gr.update(value=None),                  # temperatura
        gr.update(value=None),                  # frecuencia_respiratoria
        gr.update(visible=True),                # etapa_identificacion
        gr.update(visible=False)                # etapa_signos
    )



def registrar_paciente_y_actualizar_vistas(*args):
    """Guarda el registro y actualiza solo las vistas ligeras para agilizar el proceso."""
    resultado = list(registrar_paciente(*args))

    # Si el registro fue exitoso, asignar el siguiente consecutivo disponible.
    if resultado[1] is not None:
        resultado[2] = gr.update(value=generar_siguiente_codigo())

    # El historial y el resumen principal se actualizan inmediatamente.
    # Los indicadores se recalculan al entrar a su módulo o al pulsar Actualizar.
    tabla_actualizada = cargar_historial_vista()
    inicio_actualizado = resumen_inicio()
    return (*resultado, tabla_actualizada, inicio_actualizado)

# =====================================================
# 5. INTERFAZ PRINCIPAL - REDISEÑO VISUAL MEDICAL
# =====================================================


# =====================================================
# 5. INTERFAZ V4 - NAVEGACIÓN LATERAL Y ESTILO CLÍNICO
# =====================================================

CSS_MEDICAL = """
:root {
  --ink: #163247;
  --muted: #6b8191;
  --teal: #087f83;
  --blue: #176b87;
  --line: #dce8ee;
  --surface: #ffffff;
  --canvas: #f3f7fa;
}
html, body, .gradio-container {
  background: #f3f7fa !important;
  color: var(--ink) !important;
  font-family: Inter, ui-sans-serif, system-ui, sans-serif !important;
}
.gradio-container { max-width: 1600px !important; padding: 18px !important; }
#topbar {
  display:flex; align-items:center; justify-content:space-between; gap:18px;
  padding:20px 26px; margin-bottom:18px; border-radius:20px;
  color:#fff; background:linear-gradient(110deg,#10364f 0%,#176b87 58%,#0b8885 100%);
  box-shadow:0 12px 30px rgba(17,57,78,.14);
}
#topbar h1 { color:#fff !important; font-size:clamp(23px,2.5vw,31px); margin:3px 0 5px; }
#topbar p { color:#e5f4f6 !important; margin:0; font-size:13px; }
#topbar .eyebrow { color:#bfe8e7; font-size:10px; letter-spacing:.16em; font-weight:800; text-transform:uppercase; }
#topbar .brand-mark {
  width:52px;height:52px;border-radius:15px;background:rgba(255,255,255,.15);
  display:flex;align-items:center;justify-content:center;font-size:27px;flex-shrink:0;
}
#safety-note {
  border:1px solid #f0d59c; background:#fff8e9; color:#76551c;
  border-radius:12px; padding:11px 15px; margin-bottom:18px; font-size:12px;
}
#shell { gap:18px !important; align-items:stretch !important; }
#sidebar {
  background:#12394f !important; border:0 !important; border-radius:18px !important;
  padding:16px !important; min-width:205px; max-width:230px;
  box-shadow:0 8px 24px rgba(17,57,78,.12);
}
#sidebar h3 { color:#fff !important; font-size:12px; letter-spacing:.12em; margin:5px 8px 12px; }
#sidebar p { color:#a9c5d2 !important; font-size:11px; margin:0 8px 15px; }
#nav-menu { gap:7px !important; }
#nav-menu label { display:flex !important; align-items:center !important; width:100% !important; cursor:pointer !important; color:inherit !important; }
#nav-menu [role="radio"] {
  border:1px solid transparent !important; border-radius:11px !important;
  padding:12px 13px !important; background:transparent !important;
  color:#d9e9ef !important; transition:all .15s ease;
}
#nav-menu [role="radio"]:hover { background:rgba(255,255,255,.08) !important; }
#nav-menu [role="radio"][aria-checked="true"] {
  background:#e3f3f3 !important; color:#075f68 !important;
  border-color:#c4e4e3 !important; font-weight:750 !important;
}
#nav-menu [data-testid="radio"] { gap:10px !important; width:100% !important; }\n#nav-menu input[type="radio"] { accent-color:#0b8586 !important; }
#workspace { min-width:0; gap:16px !important; }
.view-panel {
  background:transparent !important; border:0 !important; padding:2px 2px 22px !important;
  gap:15px !important;
}
.page-title { margin:0 !important; color:#163247 !important; font-size:25px !important; }
.page-subtitle { color:#6b8191 !important; margin-top:-8px !important; font-size:13px !important; }
.surface-card {
  background:#fff !important; border:1px solid var(--line) !important;
  border-radius:16px !important; padding:18px !important;
  box-shadow:0 5px 18px rgba(20,67,88,.045) !important;
}
.kpi-card {
  background:#fff; border:1px solid var(--line); border-radius:16px;
  padding:17px 18px; min-height:112px; box-shadow:0 5px 18px rgba(20,67,88,.04);
}
.kpi-label { color:#718596; font-size:11px; font-weight:750; letter-spacing:.08em; text-transform:uppercase; }
.kpi-value { color:#163b52; font-size:29px; line-height:1.2; font-weight:800; margin-top:8px; }
.kpi-hint { color:#8294a2; font-size:11px; margin-top:5px; }
.mini-label { color:#718596; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }
#sidebar footer, footer { display:none !important; }
button, input, textarea { border-radius:10px !important; }
button.primary {
  background:linear-gradient(100deg,#176b87,#078b88) !important;
  border:0 !important; color:#fff !important; font-weight:700 !important;
}
button.primary:hover { filter:brightness(1.04); }
label { color:#36536b !important; font-weight:650 !important; }
.prose, .markdown { color:#29465a; }
table { border-collapse:collapse; }
@media (max-width: 850px) {
  .gradio-container { padding:10px !important; }
  #shell { flex-direction:column !important; }
  #sidebar { max-width:none; min-width:0; }
  #nav-menu { display:grid !important; grid-template-columns:repeat(2,minmax(0,1fr)); }
  #topbar { padding:17px; }
}

/* V5 · Vista de evaluación centrada en lectura rápida */
.clinical-results { font-family: Inter, system-ui, sans-serif; color:#17324d; }
.clinical-result-heading { display:flex; justify-content:space-between; align-items:center; gap:14px; flex-wrap:wrap; }
.clinical-result-heading h2 {font-size:25px; margin:4px 0;} .clinical-result-heading p {font-size:13px; color:#64748b; margin:0;}
.clinical-eyebrow {font-size:11px; letter-spacing:.1em; font-weight:800; color:#0f858a;}
.clinical-alert-count {padding:9px 14px; border-radius:12px; background:#fff1db; color:#87500a; font-weight:700;}
.clinical-model-grid {display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; margin:22px 0 14px;}
.clinical-model-card {border:1px solid #dbe8ed; border-radius:15px; padding:20px; background:#f5fafb; display:flex; flex-direction:column; gap:8px;}
.clinical-model-card span {font-size:12px; letter-spacing:.07em; color:#5a7081; font-weight:800;}
.clinical-model-card strong {font-size:33px; line-height:1.1; color:#123d55;}
.clinical-model-card small {font-size:12px; color:#60798b;}
.clinical-comparison {padding:13px 16px; border-radius:12px; display:flex; flex-direction:column; gap:4px; margin-bottom:22px;}
.clinical-comparison.match {background:#e7f7f0; color:#116b4a;} .clinical-comparison.mismatch {background:#fff1e4; color:#8a510d;}
.clinical-comparison span {font-size:12px;}
.clinical-results h3 {font-size:17px; margin:18px 0 12px;}
.clinical-vitals {display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px;}
.clinical-vital {border:1px solid #e0e9ee; border-radius:12px; padding:13px; background:#fff;}
.clinical-vital-label {font-size:12px; color:#61788a; margin-bottom:6px;}
.clinical-vital-value {font-size:20px; font-weight:750; color:#163b53;}
.clinical-vital-value small {font-size:12px; font-weight:500; color:#6b8291;}
.clinical-alert-box {border-radius:14px; padding:14px 18px; margin-top:22px;}
.clinical-alert-active {background:#fff8eb; border:1px solid #f3d6a6;} .clinical-alert-clear {background:#f0faf5; border:1px solid #cce9d9;}
.clinical-alert-box h3 {margin:0 0 10px;} .clinical-alert-item {font-size:14px; line-height:1.6; margin:6px 0;}
.clinical-disclaimer {font-size:12px; color:#687e8c; margin-top:18px;}
.clinical-empty {padding:20px; color:#8a510d;}
@media(max-width:760px) {.clinical-model-grid {grid-template-columns:1fr;} .clinical-vitals {grid-template-columns:repeat(2,minmax(0,1fr));}}
"""

def navegar(seccion):
    visibles = [
        seccion == "Inicio",
        seccion == "Registro",
        seccion == "Evaluación",
        seccion == "Historial",
        seccion == "Indicadores",
        seccion == "Documentación"
    ]
    return [gr.update(visible=valor) for valor in visibles]


def resumen_inicio():
    """Resume los registros usando las mismas reglas de alertas que Indicadores."""
    historial = cargar_historial()
    total = len(historial)
    pacientes = historial["codigo"].nunique() if total and "codigo" in historial else 0

    alertas = 0
    for _, fila in historial.iterrows():
        try:
            datos_alertas = {
                "edad": float(fila["edad"]),
                "spo2": float(fila["spo2"]),
                "temperatura": float(fila["temperatura"]),
                "presion_sistolica": float(fila["presion_sistolica"]),
                "presion_diastolica": float(fila["presion_diastolica"])
            }
            resultado_alertas = detectar_condiciones_criticas(datos_alertas)
            if resultado_alertas["cantidad_alertas"] > 0:
                alertas += 1
        except (ValueError, TypeError, KeyError):
            continue

    porcentaje = (alertas / total * 100) if total else 0
    return f"""
    <div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px">
    <div class="kpi-card"><div class="kpi-label">Registros simulados</div>
      <div class="kpi-value">{total}</div><div class="kpi-hint">Evaluaciones almacenadas</div></div>
    <div class="kpi-card"><div class="kpi-label">Pacientes registrados</div>
      <div class="kpi-value">{pacientes}</div><div class="kpi-hint">Códigos diferentes</div></div>
    <div class="kpi-card"><div class="kpi-label">Registros con alertas</div>
      <div class="kpi-value">{alertas}</div><div class="kpi-hint">Según reglas experimentales</div></div>
    <div class="kpi-card"><div class="kpi-label">Porcentaje con alertas</div>
      <div class="kpi-value">{porcentaje:.1f}%</div><div class="kpi-hint">Sobre registros almacenados</div></div>
    </div>
    """


with gr.Blocks(title=TITULO) as app:

    gr.HTML("""
    <header id="topbar">
      <div style="display:flex;align-items:center;gap:16px">
        <div class="brand-mark">✚</div>
        <div>
          <div class="eyebrow">Plataforma académica · Entorno de simulación</div>
          <h1>Sistema Integrado de Soporte para Triaje Clínico</h1>
          <p>Registro, evaluación experimental y análisis de información simulada</p>
        </div>
      </div>
      <div style="text-align:right;color:#d8eeee;font-size:11px;line-height:1.7">
        <strong style="color:#fff">PROTOTIPO V7.8</strong><br/>Módulos integrados
      </div>
    </header>
    <div id="safety-note">⚠️ <strong>Uso académico:</strong> utilizar exclusivamente datos ficticios. Los niveles I–V y las alertas son experimentales. Este prototipo no está validado clínicamente y no debe utilizarse para decisiones asistenciales.</div>
    """)

    with gr.Row(elem_id="shell"):
        with gr.Column(scale=1, min_width=205, elem_id="sidebar"):
            gr.HTML("<h3>NAVEGACIÓN</h3><p>Seleccione un módulo para continuar</p>")
            nav = gr.Radio(
                choices=["Inicio", "Registro", "Evaluación", "Historial", "Indicadores", "Documentación"],
                value="Inicio", show_label=False, container=False,
                elem_id="nav-menu"
            )
            gr.HTML("<div style='margin:25px 8px 5px;color:#8fb2c2;font-size:10px;line-height:1.7'>SISTEMA DE PRUEBAS<br><strong style='color:#d9edf1'>Motores independientes</strong><br>Matemático · Random Forest</div>")

        with gr.Column(scale=5, elem_id="workspace"):

            with gr.Column(visible=True, elem_id="view-home", elem_classes=["view-panel"]) as vista_inicio:
                gr.Markdown("# Centro de control", elem_classes=["page-title"])
                gr.Markdown("Resumen del entorno de pruebas y acceso a las funciones principales.", elem_classes=["page-subtitle"])
                tarjetas_inicio = gr.HTML(resumen_inicio())
                with gr.Row():
                    with gr.Column(scale=3, elem_classes=["surface-card"]):
                        gr.Markdown("### Flujo de trabajo")
                        gr.Markdown("**01 · Registro**  Captura de datos ficticios.\n\n**02 · Validación**  Comprobación de campos y límites técnicos.\n\n**03 · Procesamiento**  Normalización y cálculo del índice matemático.\n\n**04 · Comparación**  Predicción independiente de Random Forest.\n\n**05 · Alertas**  Evaluación mediante reglas separadas.\n\n**06 · Consulta**  Historial e indicadores del conjunto de pruebas.")
                    with gr.Column(scale=2, elem_classes=["surface-card"]):
                        gr.Markdown("### Arquitectura funcional")
                        gr.Markdown("""
                        <div style="padding:10px 0">
                          <div style="background:#eaf4f7;padding:12px;border-radius:10px;text-align:center"><b>Entrada de datos</b><br><small>Captura y validación</small></div>
                          <div style="text-align:center;color:#0b8586;font-size:22px">↓</div>
                          <div style="display:grid;grid-template-columns:1fr 1fr;gap:9px">
                            <div style="background:#edf3fb;border:1px solid #d9e5f2;padding:12px;border-radius:10px;text-align:center"><b>Modelo matemático</b><br><small>Índice ponderado</small></div>
                            <div style="background:#e8f6f3;border:1px solid #d0e9e2;padding:12px;border-radius:10px;text-align:center"><b>Random Forest</b><br><small>Predicción IA</small></div>
                          </div>
                          <div style="text-align:center;color:#0b8586;font-size:22px">↓</div>
                          <div style="background:#fff5e7;border:1px solid #f2dfbd;padding:12px;border-radius:10px;text-align:center"><b>Motor de alertas independiente</b></div>
                          <div style="text-align:center;color:#0b8586;font-size:22px">↓</div>
                          <div style="background:#f0f4f7;padding:12px;border-radius:10px;text-align:center"><b>Historial e indicadores</b></div>
                        </div>
                        """)
                boton_actualizar_inicio = gr.Button("Actualizar resumen", variant="secondary")
                boton_actualizar_inicio.click(fn=resumen_inicio, inputs=[], outputs=tarjetas_inicio)

            with gr.Column(visible=False, elem_id="view-register", elem_classes=["view-panel"]) as vista_registro:
                gr.Markdown("# ◉ Registro de paciente simulado", elem_classes=["page-title"])
                gr.Markdown("Proceso de captura en dos etapas: identificación e información de ingreso, seguido de signos y variables fisiológicas.", elem_classes=["page-subtitle"])

                with gr.Group(visible=True, elem_classes=["surface-card"]) as etapa_identificacion:
                    gr.Markdown("### 👤 ① Identificación e información de ingreso")
                    gr.Markdown("Utilice exclusivamente identidades y datos ficticios. No registre información de personas reales.")
                    with gr.Row():
                        codigo = gr.Textbox(label="🆔 Código interno (automático)", placeholder="Se asigna automáticamente", interactive=False)
                        tipo_documento = gr.Dropdown(
                            choices=["Documento ficticio", "Cédula ficticia", "Pasaporte ficticio", "Otro identificador ficticio"],
                            value="Documento ficticio", label="📄 Tipo de identificación"
                        )
                        documento = gr.Textbox(label="🔖 Número / identificador ficticio", placeholder="FICTICIO-0010")
                    with gr.Row():
                        nombres = gr.Textbox(label="👤 Nombres ficticios", placeholder="Ejemplo: Juan Carlos")
                        apellidos = gr.Textbox(label="👤 Apellidos ficticios", placeholder="Ejemplo: Pérez Gómez")
                    with gr.Row():
                        edad = gr.Number(label="🎂 Edad (años)", precision=0)
                        sexo = gr.Dropdown(choices=["Femenino", "Masculino", "Intersexual", "No especificado"], label="⚧ Sexo registrado")
                    motivo_consulta = gr.Textbox(label="🩺 Motivo de consulta", placeholder="Describa brevemente el motivo del ingreso", lines=2)
                    sintomas = gr.Textbox(label="💬 Síntomas principales", placeholder="Ejemplo: dolor de cabeza, mareo, tos...", lines=3)
                    observaciones = gr.Textbox(label="📝 Observaciones adicionales (opcional)", placeholder="Información contextual del caso simulado", lines=2)
                    boton_continuar = gr.Button("Continuar a signos y variables  →", variant="primary", size="lg")
                    estado_etapa = gr.Markdown()

                with gr.Group(visible=False, elem_classes=["surface-card"]) as etapa_signos:
                    gr.Markdown("### ♥ ② Captura de signos y variables")
                    gr.Markdown("Registre las variables que alimentan los motores matemático y Random Forest. Los límites son técnicos y experimentales.")
                    with gr.Row():
                        dolor = gr.Number(label="⚡ Nivel de dolor (0–10)", precision=0)
                        fc = gr.Number(label="❤️ Frecuencia cardíaca (bpm)", precision=0)
                    with gr.Row():
                        spo2 = gr.Number(label="🫁 Saturación de oxígeno (%)", precision=0)
                        temperatura = gr.Number(label="🌡️ Temperatura (°C)", precision=1)
                    with gr.Row():
                        sistolica = gr.Number(label="🩸 Presión sistólica (mmHg)", precision=0)
                        diastolica = gr.Number(label="🩸 Presión diastólica (mmHg)", precision=0)
                    fr = gr.Number(label="🌬️ Frecuencia respiratoria (rpm)", precision=0)
                    with gr.Row():
                        boton_volver = gr.Button("← Volver a identificación", variant="secondary")
                        boton_guardar = gr.Button("Validar y guardar registro", variant="primary", size="lg")

                resultado = gr.Markdown(elem_classes=["surface-card"])
                tabla = gr.Dataframe(label="Resumen del registro guardado", interactive=False, wrap=True)

                boton_continuar.click(
                    fn=validar_etapa_identificacion,
                    inputs=[codigo, tipo_documento, documento, nombres, apellidos, edad, sexo],
                    outputs=[estado_etapa, etapa_identificacion, etapa_signos]
                )
                boton_volver.click(
                    fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
                    inputs=[], outputs=[etapa_identificacion, etapa_signos]
                )

            with gr.Column(visible=False, elem_id="view-evaluation", elem_classes=["view-panel"]) as vista_evaluacion:
                gr.Markdown("# Evaluación comparativa", elem_classes=["page-title"])
                gr.Markdown("Consulte un registro y compare los dos motores. Las alertas se calculan mediante reglas independientes.", elem_classes=["page-subtitle"])
                with gr.Row():
                    with gr.Column(scale=3, elem_classes=["surface-card"]):
                        codigo_evaluacion = gr.Textbox(label="Código del paciente", placeholder="Ejemplo: SIM-009")
                        boton_evaluar = gr.Button("Procesar evaluación", variant="primary", size="lg")
                    with gr.Column(scale=2, elem_classes=["surface-card"]):
                        gr.Markdown("### Motores independientes")
                        gr.Markdown("**Modelo matemático:** conserva el índice ponderado original.\n\n**Random Forest:** genera su propia predicción sobre las variables normalizadas.\n\n**Alertas:** aplica reglas discretas sin depender de los niveles.")
                resultado_evaluacion = gr.HTML(elem_classes=["surface-card"])
                with gr.Accordion("Ver detalles técnicos del procesamiento (uso académico)", open=False):
                    detalle_evaluacion = gr.Markdown(elem_classes=["surface-card"])
                boton_evaluar.click(fn=evaluar_paciente, inputs=[codigo_evaluacion], outputs=[resultado_evaluacion, detalle_evaluacion])

            with gr.Column(visible=False, elem_id="view-history", elem_classes=["view-panel"]) as vista_historial:
                gr.Markdown("# Historial de registros", elem_classes=["page-title"])
                gr.Markdown("Consulte los registros simulados y abra la ficha individual de cada caso.", elem_classes=["page-subtitle"])
                with gr.Row(elem_classes=["surface-card"]):
                    boton_historial = gr.Button("↻ Actualizar historial", variant="primary")
                tabla_historial = gr.Dataframe(
                    value=cargar_historial_vista(),
                    label="Registros almacenados",
                    interactive=False,
                    wrap=False,
                    max_height=520,
                    column_widths=["15%", "8%", "15%", "14%", "5%", "17%", "10%", "10%", "6%"]
                )
                boton_historial.click(fn=cargar_historial_vista, inputs=[], outputs=tabla_historial)

                with gr.Group(elem_classes=["surface-card"]):
                    gr.Markdown("### 🔎 Consultar ficha individual")
                    with gr.Row():
                        codigo_ficha = gr.Textbox(label="Código del paciente", placeholder="Ejemplo: SIM-012")
                        boton_ficha = gr.Button("Ver ficha completa", variant="primary")
                    ficha_historial = gr.HTML()
                    boton_ficha.click(
                        fn=consultar_ficha_historial,
                        inputs=[codigo_ficha],
                        outputs=[ficha_historial]
                    )

            with gr.Column(visible=False, elem_id="view-indicators", elem_classes=["view-panel"]) as vista_indicadores:
                gr.Markdown("# Indicadores y análisis", elem_classes=["page-title"])
                gr.Markdown("Estadísticas descriptivas calculadas sobre los registros simulados.", elem_classes=["page-subtitle"])
                boton_indicadores = gr.Button("Actualizar indicadores", variant="primary")
                resumen_indicadores = gr.Markdown(elem_classes=["surface-card"])
                with gr.Row():
                    with gr.Column(elem_classes=["surface-card"]):
                        grafico_fechas = gr.Plot(label="Registros por fecha")
                    with gr.Column(elem_classes=["surface-card"]):
                        grafico_alertas = gr.Plot(label="Distribución de alertas")
                with gr.Row():
                    with gr.Column(elem_classes=["surface-card"]):
                        grafico_alertas_edad = gr.Plot(label="Alertas por grupo de edad")
                    with gr.Column(elem_classes=["surface-card"]):
                        grafico_niveles_edad = gr.Plot(label="Niveles experimentales por edad")
                boton_indicadores.click(fn=generar_indicadores, inputs=[],
                    outputs=[resumen_indicadores, grafico_fechas, grafico_alertas, grafico_alertas_edad, grafico_niveles_edad])

            with gr.Column(visible=False, elem_id="view-doc", elem_classes=["view-panel"]) as vista_documentacion:

                gr.Markdown("""
                ## Documentación técnica del prototipo

                Esta sección describe los criterios matemáticos y funcionales
                implementados en el Sistema Integrado de Soporte para Triaje Clínico.
                """)

                gr.Markdown("""
                ### 1. Propósito y alcance

                El sistema es un **prototipo académico** para registrar, validar y
                procesar datos ficticios mediante reglas informáticas y un modelo
                matemático ponderado.

                **No es un dispositivo médico, no cuenta con validación clínica y no
                debe utilizarse para diagnosticar, clasificar ni tomar decisiones sobre
                pacientes reales.** Los resultados I–V son niveles matemáticos
                experimentales y no equivalen a las categorías oficiales de triaje.
                """)

                gr.Markdown("""
                ### 2. Variables de entrada

                | Variable | Unidad / escala | Uso |
                |---|---|---|
                | Edad | Años | Transformación experimental de riesgo |
                | Nivel de dolor | 0–10 | Normalización directa |
                | Frecuencia cardíaca | bpm | Normalización y regla experimental |
                | Saturación de oxígeno | % | Normalización inversa orientada al riesgo |
                | Presión sistólica | mmHg | Cálculo de PAM |
                | Presión diastólica | mmHg | Cálculo de PAM |
                | Temperatura | °C | Normalización y regla experimental |
                | Frecuencia respiratoria | rpm | Normalización y regla experimental |

                Los límites de captura son límites técnicos del software, no rangos
                de referencia clínica.
                """)

                gr.Markdown("""
                ### 3. Ponderación del modelo matemático

                | Variable | Peso |
                |---|---:|
                | Saturación de oxígeno (SpO₂) | 0.25 |
                | Presión arterial media (PAM) | 0.20 |
                | Frecuencia cardíaca | 0.15 |
                | Frecuencia respiratoria | 0.15 |
                | Temperatura | 0.10 |
                | Nivel de dolor | 0.10 |
                | Edad | 0.05 |
                | **Total** | **1.00** |

                **Fórmula:**

                `Ip = (0.25 × SpO₂_riesgo) + (0.20 × PAM_riesgo) + (0.15 × FC_riesgo) + (0.15 × FR_riesgo) + (0.10 × Temp_riesgo) + (0.10 × Dolor_norm) + (0.05 × Edad_riesgo)`

                Cada contribución es el valor orientado al riesgo multiplicado por
                su peso. El índice final se encuentra entre 0 y 1.
                """)

                gr.Markdown("""
                ### 4. Normalización y orientación al riesgo

                Para las variables normalizadas mediante rango se utiliza:

                `X_norm = (X - X_min) / (X_max - X_min)`

                | Variable | Rango | Criterio implementado |
                |---|---:|---|
                | Edad | 0–120 años | Menor de 65 = 0; 65–79 = 0.5; 80 o más = 1 |
                | Dolor | 0–10 | Normalización directa |
                | Frecuencia cardíaca | 30–220 bpm | Regla por intervalos de riesgo |
                | Frecuencia respiratoria | 5–50 rpm | Regla por intervalos de riesgo |
                | Temperatura | 30–42 °C | Regla por intervalos de riesgo |
                | SpO₂ | 85–100 % | Riesgo = 1 − valor normalizado |
                | PAM | Derivada de PAS y PAD | Regla por intervalos de riesgo |

                **Reglas experimentales programadas:**

                - Frecuencia cardíaca: 51–90 = 0; 41–50 o 91–110 = 0.25;
                  111–130 = 0.50; los demás valores admitidos = 1.
                - Frecuencia respiratoria: 12–20 = 0; 9–11 = 0.25;
                  21–24 = 0.50; los demás valores admitidos = 1.
                - Temperatura: 36.1–38.0 °C = 0; 35.1–36.0 °C o más de 38.0
                  hasta 39.0 °C = 0.25; más de 39.0 hasta 41.0 °C = 0.50;
                  los demás valores admitidos = 1.
                - SpO₂: se invierte la normalización para que un valor menor
                  produzca un valor de riesgo mayor.
                - Edad: se aplica la transformación por grupos indicada arriba.
                - Dolor: se utiliza el valor normalizado entre 0 y 1.

                Estas reglas son decisiones de implementación; no son escalas clínicas
                validadas.
                """)

                gr.Markdown("""
                ### 5. Presión arterial media (PAM)

                `PAM = (PAS + 2 × PAD) / 3`

                `PAM_norm = limitar((PAM - 60) / 40, 0, 1)`

                | PAM calculada | Valor de riesgo |
                |---|---:|
                | Menor de 60 mmHg | 1.00 |
                | Desde 60 y menor de 65 mmHg | 0.50 |
                | Desde 65 hasta 100 mmHg | 0.00 |
                | Mayor de 100 mmHg | 0.25 |

                El cálculo y los intervalos se utilizan únicamente para demostrar
                el procesamiento matemático implementado.
                """)

                gr.Markdown("""
                ### 6. Niveles del índice experimental

                | Nivel | Intervalo del índice |
                |---|---:|
                | I | 0.80 a 1.00 |
                | II | 0.60 a menos de 0.80 |
                | III | 0.40 a menos de 0.60 |
                | IV | 0.20 a menos de 0.40 |
                | V | 0.00 a menos de 0.20 |

                Estos niveles son intervalos matemáticos del prototipo. No equivalen
                a las categorías oficiales de triaje colombiano ni determinan la
                prioridad real de atención.
                """)

                gr.Markdown("""
                ### 7. Criterios de generación de alertas

                | Variable / condición | Criterio implementado |
                |---|---|
                | Saturación de oxígeno | SpO₂ menor de 85 %: alerta |
                | Temperatura | Mayor de 42 °C: alerta |
                | PAM | Menor de 60 mmHg: alerta roja |
                | PAM | Desde 60 y menor de 65 mmHg: alerta amarilla |
                | PAM | Mayor de 100 mmHg: alerta amarilla |
                | PAM | Desde 65 hasta 100 mmHg: estado verde |

                El índice ponderado y las alertas son mecanismos diferentes. Por eso,
                un registro puede obtener un nivel experimental y no generar alertas
                discretas.
                """)

                gr.Markdown("""
                ### 8. Indicadores y análisis por edad

                El panel presenta registros totales, pacientes diferentes, registros
                con y sin alertas, porcentaje general de alertas, registros por fecha
                y distribución de alertas.

                Grupos de edad utilizados:

                - 0–17 años
                - 18–39 años
                - 40–59 años
                - 60–79 años
                - 80 años o más

                El porcentaje de alertas de cada grupo se calcula dividiendo los
                registros con alertas entre el total de registros válidos del grupo
                y multiplicando por 100. Los resultados describen únicamente los
                registros simulados; no demuestran asociación clínica entre edad y riesgo.
                """)

                gr.Markdown("""
                ### 9. Motor experimental de inteligencia artificial

                El prototipo incorpora un segundo motor basado en Random Forest.
                Este recibe las mismas siete variables normalizadas utilizadas por
                el modelo matemático original y genera una predicción independiente
                del nivel experimental I–V. La interfaz presenta ambos resultados
                y señala si coinciden o difieren. Random Forest no reemplaza ni
                modifica el índice matemático original.

                Las probabilidades por clase que pueda mostrar el clasificador son
                salidas internas del modelo entrenado con datos sintéticos; no son
                probabilidades clínicas. La regresión lineal se conserva como
                herramienta de análisis comparativo fuera del flujo principal.

                ### 10. Flujo funcional

                1. **Registro:** captura de un código y variables simuladas.
                2. **Validación:** revisión de campos y límites técnicos.
                3. **Normalización:** conversión de variables a escalas de procesamiento.
                4. **Procesamiento:** cálculo de PAM, orientación al riesgo e índice ponderado.
                5. **Alertas:** evaluación de las condiciones discretas programadas.
                6. **Almacenamiento:** conservación del registro en el historial CSV.
                7. **Consulta:** presentación del resultado e indicadores experimentales.
                """)

                gr.Markdown("""
                ### 11. Limitaciones y consideraciones

                - El prototipo trabaja con datos ficticios y pruebas académicas.\n            - Random Forest aprende a reproducir las etiquetas sintéticas del modelo matemático; no tiene validación clínica.
                - Las ponderaciones, rangos y reglas de riesgo no han sido validados clínicamente.
                - Los niveles I–V no sustituyen las categorías oficiales de triaje.
                - La transformación de edad es experimental y requiere revisión específica
                  antes de considerar población pediátrica.
                - Los indicadores dependen de la cantidad y composición de los registros simulados.
                - No se ha implementado integración real con dispositivos médicos,
                  historias clínicas ni estándares HL7/FHIR.
                - Un uso futuro en entornos asistenciales exigiría validación, controles
                  de seguridad y supervisión profesional.

                **Uso permitido:** demostración académica, pruebas de software y análisis
                matemático con datos simulados.
                """)


    # Se enlaza el evento después de crear todas las vistas y componentes
    # que reciben actualización tras guardar un registro.
    boton_guardar.click(
        fn=registrar_paciente_y_actualizar_vistas,
        inputs=[codigo, tipo_documento, documento, nombres, apellidos, edad, sexo,
                motivo_consulta, sintomas, observaciones, dolor, fc, spo2,
                sistolica, diastolica, temperatura, fr],
        outputs=[
            resultado, tabla,
            codigo, tipo_documento, documento, nombres, apellidos, edad, sexo,
            motivo_consulta, sintomas, observaciones,
            dolor, fc, spo2, sistolica, diastolica, temperatura, fr,
            etapa_identificacion, etapa_signos,
            tabla_historial, tarjetas_inicio
        ]
    )

    vistas = [vista_inicio, vista_registro, vista_evaluacion, vista_historial, vista_indicadores, vista_documentacion]
    # Asignar el primer código disponible al abrir el formulario.
    app.load(fn=generar_siguiente_codigo, inputs=[], outputs=[codigo])
    nav.change(fn=navegar, inputs=nav, outputs=vistas)

if __name__ == "__main__":
    app.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.teal,
            secondary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.slate,
            radius_size="lg",
            spacing_size="md"
        ),
        css=CSS_MEDICAL
    )