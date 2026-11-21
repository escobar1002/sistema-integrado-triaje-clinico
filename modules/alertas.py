
# =====================================================
# MODULO DE DETECCION DE CONDICIONES CRITICAS
# Sistema Integrado de Soporte para Triaje Clinico
# =====================================================

from modules.presion_arterial import calcular_pam


def detectar_condiciones_criticas(datos):
    """
    Identifica condiciones que cumplen los umbrales
    de alerta propuestos para el prototipo.

    Recibe los valores originales, antes de normalizarlos.

    IMPORTANTE:
    Estos criterios son parametros academicos de diseño.
    No representan una evaluacion clinica validada.
    """

    alertas = []

    spo2 = datos.get("spo2")
    temperatura = datos.get("temperatura")

    # ---------------------------------------------
    # 1. Alerta por saturacion de oxigeno
    # ---------------------------------------------

    if spo2 is not None and spo2 < 85:

        alertas.append({
            "variable": "Saturacion de oxigeno",
            "valor": f"{spo2} %",
            "criterio": "Inferior al 85 %",
            "nivel": "ROJO",
            "mensaje": (
                "Se cumple el criterio de alerta "
                "de saturacion de oxigeno."
            )
        })

    # ---------------------------------------------
    # 2. Alerta por temperatura
    # ---------------------------------------------

    if temperatura is not None and temperatura > 42:

        alertas.append({
            "variable": "Temperatura",
            "valor": f"{temperatura} °C",
            "criterio": "Superior a 42 °C",
            "nivel": "ROJO",
            "mensaje": (
                "Se cumple el criterio de alerta "
                "de temperatura."
            )
        })

    # ---------------------------------------------
    # 3. Evaluacion de presion arterial media (PAM)
    # Adaptacion individual del prototipo
    # ---------------------------------------------

    presion_sistolica = datos.get("presion_sistolica")
    presion_diastolica = datos.get("presion_diastolica")

    evaluacion_pam = {
        "pam": None,
        "estado": "PENDIENTE",
        "nivel": "GRIS",
        "mensaje": (
            "No se puede calcular la PAM: "
            "faltan los valores de presion arterial."
        )
    }

    if (
        presion_sistolica is not None
        and presion_diastolica is not None
    ):

        pam = calcular_pam(
            presion_sistolica,
            presion_diastolica
        )

        # PAM inferior a 60 mmHg
        if pam < 60:

            evaluacion_pam = {
                "pam": pam,
                "estado": "PAM MUY BAJA",
                "nivel": "ROJO",
                "mensaje": (
                    "La PAM esta por debajo de 60 mmHg. "
                    "Se cumple el criterio de alerta baja."
                )
            }

            alertas.append({
                "variable": "Presion arterial media",
                "valor": f"{pam} mmHg",
                "criterio": "Inferior a 60 mmHg",
                "nivel": "ROJO",
                "mensaje": evaluacion_pam["mensaje"]
            })

        # PAM entre 60 y menos de 65 mmHg
        elif pam < 65:

            evaluacion_pam = {
                "pam": pam,
                "estado": "PRECAUCION: PAM BAJA",
                "nivel": "AMARILLO",
                "mensaje": (
                    "La PAM esta entre 60 y menos de "
                    "65 mmHg. Requiere revision."
                )
            }

            alertas.append({
                "variable": "Presion arterial media",
                "valor": f"{pam} mmHg",
                "criterio": "Entre 60 y menos de 65 mmHg",
                "nivel": "AMARILLO",
                "mensaje": evaluacion_pam["mensaje"]
            })

        # PAM entre 65 y 100 mmHg
        elif pam <= 100:

            evaluacion_pam = {
                "pam": pam,
                "estado": "DENTRO DEL INTERVALO DE REFERENCIA",
                "nivel": "VERDE",
                "mensaje": (
                    "La PAM se encuentra dentro del "
                    "intervalo de referencia del prototipo."
                )
            }

        # PAM superior a 100 mmHg
        else:

            evaluacion_pam = {
                "pam": pam,
                "estado": "PAM ELEVADA",
                "nivel": "AMARILLO",
                "mensaje": (
                    "La PAM supera los 100 mmHg. "
                    "Se recomienda revision por personal de salud."
                )
            }

            alertas.append({
                "variable": "Presion arterial media",
                "valor": f"{pam} mmHg",
                "criterio": "Superior a 100 mmHg",
                "nivel": "AMARILLO",
                "mensaje": evaluacion_pam["mensaje"]
            })

    # ---------------------------------------------
    # 4. Resultado general de la deteccion
    # ---------------------------------------------

    if alertas:
        estado = "CONDICIONES DE ALERTA DETECTADAS"
    else:
        estado = "SIN ALERTAS SEGUN LOS CRITERIOS IMPLEMENTADOS"

    return {
        "estado": estado,
        "cantidad_alertas": len(alertas),
        "alertas": alertas,
        "presion_arterial": evaluacion_pam["mensaje"],
        "evaluacion_pam": evaluacion_pam
    }
