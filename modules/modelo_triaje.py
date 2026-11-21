
# =====================================================
# MODULO DE MODELO MATEMATICO DE TRIAJE
# Sistema Integrado de Soporte para Triaje Clinico
# Proyecto de Ingenieria I - Etapa 4
# =====================================================


# =====================================================
# 1. COEFICIENTES DEL MODELO MATEMATICO
# =====================================================

# Coeficientes establecidos en la Etapa 3

COEFICIENTES = {
    "spo2": 0.25,
    "pa": 0.20,
    "frecuencia_cardiaca": 0.15,
    "frecuencia_respiratoria": 0.15,
    "temperatura": 0.10,
    "dolor": 0.10,
    "edad": 0.05
}


# =====================================================
# 2. VALIDACION DE VARIABLES
# =====================================================

def validar_variables(variables):
    """
    Verifica que todas las variables requeridas
    estén presentes y tengan valores numericos
    entre 0 y 1.
    """

    if not isinstance(variables, dict):
        raise ValueError("Las variables deben proporcionarse en un diccionario.")

    faltantes = [
        variable
        for variable in COEFICIENTES
        if variable not in variables
    ]

    if faltantes:
        raise ValueError(
            "Faltan variables para calcular el indice: "
            + ", ".join(faltantes)
        )

    for variable in COEFICIENTES:

        valor = variables[variable]

        if (
            valor is None
            or not isinstance(valor, (int, float))
            or isinstance(valor, bool)
        ):
            raise ValueError(
                f"La variable {variable} debe ser numerica."
            )

        if not 0 <= valor <= 1:
            raise ValueError(
                f"La variable {variable} debe estar entre 0 y 1."
            )


# =====================================================
# 3. CALCULO DEL INDICE PONDERADO
# =====================================================

def calcular_indice_prioridad(variables):
    """
    Calcula el indice ponderado experimental de prioridad.

    Todas las variables deben estar previamente normalizadas
    y orientadas hacia el riesgo.

    Correspondencia esperada:
    - spo2: spo2_riesgo
    - pa: pam_riesgo
    - frecuencia_cardiaca: frecuencia_cardiaca_riesgo
    - frecuencia_respiratoria: frecuencia_respiratoria_riesgo
    - temperatura: temperatura_riesgo
    - dolor: dolor normalizado
    - edad: edad_riesgo

    El modelo es experimental y no constituye una
    clasificación clínica validada.
    """

    validar_variables(variables)

    contribuciones = {}

    for variable, coeficiente in COEFICIENTES.items():

        contribuciones[variable] = (
            variables[variable] * coeficiente
        )

    indice = sum(contribuciones.values())

    return {
        "indice_prioridad": round(indice, 4),

        "contribuciones": {
            variable: round(valor, 4)
            for variable, valor in contribuciones.items()
        },

        "coeficientes": COEFICIENTES.copy()
    }


# =====================================================
# 4. CLASIFICACION EXPERIMENTAL DEL INDICE
# =====================================================

def clasificar_indice_prioridad(indice):
    """
    Clasifica el resultado del indice matematico
    utilizando los intervalos experimentales
    propuestos en la Etapa 3.

    Nivel I:   0.80 a 1.00
    Nivel II:  0.60 a menor de 0.80
    Nivel III: 0.40 a menor de 0.60
    Nivel IV:  0.20 a menor de 0.40
    Nivel V:   0.00 a menor de 0.20

    Esta clasificacion es exclusivamente experimental.
    No sustituye el triaje realizado por profesionales.
    """

    if (
        not isinstance(indice, (int, float))
        or isinstance(indice, bool)
        or not 0 <= indice <= 1
    ):
        raise ValueError(
            "El indice debe ser numerico y estar entre 0 y 1."
        )

    if indice >= 0.80:
        return "Nivel I"

    elif indice >= 0.60:
        return "Nivel II"

    elif indice >= 0.40:
        return "Nivel III"

    elif indice >= 0.20:
        return "Nivel IV"

    else:
        return "Nivel V"


# =====================================================
# 5. EVALUACION COMPLETA DEL MODELO
# =====================================================

def evaluar_modelo_triaje(variables):
    """
    Ejecuta el calculo del indice y obtiene
    su correspondiente nivel experimental.
    """

    resultado = calcular_indice_prioridad(variables)

    nivel = clasificar_indice_prioridad(
        resultado["indice_prioridad"]
    )

    resultado["nivel_experimental"] = nivel

    return resultado
