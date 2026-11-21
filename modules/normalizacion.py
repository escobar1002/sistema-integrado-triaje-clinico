
# ============================================================
# MODULO DE NORMALIZACION DE VARIABLES CLINICAS
# Sistema Integrado de Soporte para Triaje Clinico
# Proyecto de Ingenieria I - Etapa 4
# ============================================================

from modules.presion_arterial import calcular_pam


# ============================================================
# 1. NORMALIZACION GENERAL DE VARIABLES
# ============================================================

def normalizar_variable(valor, minimo, maximo):
    """
    Normaliza una variable mediante el metodo min-max
    en una escala entre 0 y 1.
    """

    if valor is None:
        raise ValueError("La variable no puede estar vacia.")

    if not isinstance(valor, (int, float)) or isinstance(valor, bool):
        raise ValueError("La variable debe ser numerica.")

    if minimo >= maximo:
        raise ValueError("El minimo debe ser menor que el maximo.")

    if not minimo <= valor <= maximo:
        raise ValueError(
            f"El valor debe estar entre {minimo} y {maximo}."
        )

    return (valor - minimo) / (maximo - minimo)


# ============================================================
# 2. TRANSFORMACION EXPERIMENTAL DE LA EDAD
# ============================================================

def transformar_edad_riesgo(edad):
    """
    Transforma la edad en una puntuacion experimental.

    Menores de 65 años: 0.00
    De 65 a 79 años:    0.50
    80 años o mas:      1.00

    No representa una escala clinica validada.
    """

    if edad is None:
        raise ValueError("La edad es obligatoria.")

    if not isinstance(edad, (int, float)) or isinstance(edad, bool):
        raise ValueError("La edad debe ser numerica.")

    if not 0 <= edad <= 120:
        raise ValueError("La edad debe estar entre 0 y 120 años.")

    if edad < 65:
        return 0.00

    elif edad < 80:
        return 0.50

    else:
        return 1.00


# ============================================================
# 3. ORIENTACION EXPERIMENTAL DE FRECUENCIA CARDIACA
# ============================================================

def calcular_riesgo_frecuencia_cardiaca(valor):
    """
    Puntuacion experimental por intervalos de frecuencia
    cardiaca (latidos por minuto).

    51-90:       0.00
    41-50:       0.25
    91-110:      0.25
    111-130:     0.50
    30-40:       1.00
    131-220:     1.00
    """

    normalizar_variable(valor, 30, 220)

    if 51 <= valor <= 90:
        return 0.00

    elif 41 <= valor <= 50 or 91 <= valor <= 110:
        return 0.25

    elif 111 <= valor <= 130:
        return 0.50

    else:
        return 1.00


# ============================================================
# 4. ORIENTACION EXPERIMENTAL DE FRECUENCIA RESPIRATORIA
# ============================================================

def calcular_riesgo_frecuencia_respiratoria(valor):
    """
    Puntuacion experimental por intervalos de frecuencia
    respiratoria (respiraciones por minuto).

    12-20:       0.00
    9-11:        0.25
    21-24:       0.50
    5-8:         1.00
    25-50:       1.00
    """

    normalizar_variable(valor, 5, 50)

    if 12 <= valor <= 20:
        return 0.00

    elif 9 <= valor <= 11:
        return 0.25

    elif 21 <= valor <= 24:
        return 0.50

    else:
        return 1.00


# ============================================================
# 5. ORIENTACION EXPERIMENTAL DE TEMPERATURA
# ============================================================

def calcular_riesgo_temperatura(valor):
    """
    Puntuacion experimental por intervalos de temperatura.

    36.1-38.0:   0.00
    35.1-36.0:   0.25
    38.1-39.0:   0.25
    39.1-41.0:   0.50
    30.0-35.0:   1.00
    Mayor de 41: 1.00
    """

    normalizar_variable(valor, 30, 42)

    if 36.1 <= valor <= 38.0:
        return 0.00

    elif 35.1 <= valor <= 36.0:
        return 0.25

    elif 38.0 < valor <= 39.0:
        return 0.25

    elif 39.0 < valor <= 41.0:
        return 0.50

    else:
        return 1.00


# ============================================================
# 6. NORMALIZACION DE PRESION ARTERIAL MEDIA (PAM)
# ============================================================

def normalizar_pam(pam):
    """
    Conserva la normalizacion min-max experimental
    de PAM en el intervalo de 60 a 100 mmHg.

    La escala se limita entre 0 y 1.
    """

    pam_normalizada = (pam - 60) / (100 - 60)

    return max(0, min(1, pam_normalizada))


def calcular_riesgo_pam(pam):
    """
    Puntuacion experimental de desviacion de PAM.

    Menor de 60 mmHg: 1.00
    60 a menor de 65: 0.50
    65 a 100:         0.00
    Mayor de 100:     0.25

    Regla academica interna, no escala clinica validada.
    """

    if not isinstance(pam, (int, float)) or isinstance(pam, bool):
        raise ValueError("La PAM debe ser numerica.")

    if pam < 0:
        raise ValueError("La PAM no puede ser negativa.")

    if pam < 60:
        return 1.00

    elif pam < 65:
        return 0.50

    elif pam <= 100:
        return 0.00

    else:
        return 0.25


# ============================================================
# 7. NORMALIZACION INTEGRAL DEL PACIENTE
# ============================================================

def normalizar_paciente(datos):
    """
    Procesa las variables registradas del paciente.

    Genera:
    - Variables normalizadas originales.
    - Puntuaciones experimentales de riesgo.
    - Transformacion experimental de edad.
    - Calculo de PAM cuando existen ambas presiones.

    Las puntuaciones no constituyen una clasificacion
    clinica validada ni sustituyen la valoracion profesional.
    """

    # --------------------------------------------------------
    # EDAD
    # --------------------------------------------------------

    edad = normalizar_variable(
        datos["edad"], 0, 120
    )

    edad_riesgo = transformar_edad_riesgo(
        datos["edad"]
    )

    # --------------------------------------------------------
    # DOLOR
    # --------------------------------------------------------

    dolor = normalizar_variable(
        datos["dolor"], 0, 10
    )

    # --------------------------------------------------------
    # FRECUENCIA CARDIACA
    # --------------------------------------------------------

    frecuencia_cardiaca = normalizar_variable(
        datos["frecuencia_cardiaca"], 30, 220
    )

    frecuencia_cardiaca_riesgo = (
        calcular_riesgo_frecuencia_cardiaca(
            datos["frecuencia_cardiaca"]
        )
    )

    # --------------------------------------------------------
    # FRECUENCIA RESPIRATORIA
    # --------------------------------------------------------

    frecuencia_respiratoria = normalizar_variable(
        datos["frecuencia_respiratoria"], 5, 50
    )

    frecuencia_respiratoria_riesgo = (
        calcular_riesgo_frecuencia_respiratoria(
            datos["frecuencia_respiratoria"]
        )
    )

    # --------------------------------------------------------
    # TEMPERATURA
    # --------------------------------------------------------

    temperatura = normalizar_variable(
        datos["temperatura"], 30, 42
    )

    temperatura_riesgo = calcular_riesgo_temperatura(
        datos["temperatura"]
    )

    # --------------------------------------------------------
    # SATURACION DE OXIGENO (SpO2)
    # --------------------------------------------------------

    spo2_normalizada = normalizar_variable(
        datos["spo2"], 85, 100
    )

    # Inversion de escala: menor saturacion,
    # mayor puntuacion experimental de riesgo.
    spo2_riesgo = 1 - spo2_normalizada

    # --------------------------------------------------------
    # PRESION ARTERIAL MEDIA (PAM)
    # --------------------------------------------------------

    pam = None
    pam_normalizada = None
    pam_riesgo = None

    if (
        "presion_sistolica" in datos
        and "presion_diastolica" in datos
    ):

        pam = calcular_pam(
            datos["presion_sistolica"],
            datos["presion_diastolica"]
        )

        pam_normalizada = normalizar_pam(pam)

        pam_riesgo = calcular_riesgo_pam(pam)

    # --------------------------------------------------------
    # RESULTADO DE NORMALIZACION
    # --------------------------------------------------------

    resultado = {

        # Edad
        "edad": round(edad, 4),
        "edad_riesgo": round(edad_riesgo, 4),

        # Dolor
        "dolor": round(dolor, 4),

        # Frecuencia cardiaca
        "frecuencia_cardiaca": round(
            frecuencia_cardiaca, 4
        ),

        "frecuencia_cardiaca_riesgo": round(
            frecuencia_cardiaca_riesgo, 4
        ),

        # Frecuencia respiratoria
        "frecuencia_respiratoria": round(
            frecuencia_respiratoria, 4
        ),

        "frecuencia_respiratoria_riesgo": round(
            frecuencia_respiratoria_riesgo, 4
        ),

        # Temperatura
        "temperatura": round(temperatura, 4),

        "temperatura_riesgo": round(
            temperatura_riesgo, 4
        ),

        # Saturacion de oxigeno
        "spo2_normalizada": round(
            spo2_normalizada, 4
        ),

        "spo2_riesgo": round(
            spo2_riesgo, 4
        )
    }

    # Agregar PAM cuando existan ambas presiones
    if pam is not None:

        resultado["pam"] = pam

        resultado["pam_normalizada"] = round(
            pam_normalizada, 4
        )

        resultado["pam_riesgo"] = round(
            pam_riesgo, 4
        )

    return resultado
