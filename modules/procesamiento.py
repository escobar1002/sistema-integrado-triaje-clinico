
# modules/procesamiento.py

"""
Validación informática de datos para un prototipo académico.
Los límites técnicos no representan rangos clínicos de normalidad.
"""

RANGOS_DISENO = {
    "edad": (0, 120),
    "dolor": (0, 10),
    "frecuencia_cardiaca": (30, 220),
    "spo2": (85, 100),
    "presion_sistolica": (1, 300),
    "presion_diastolica": (1, 200),
    "temperatura": (30, 42),
    "frecuencia_respiratoria": (5, 50),
}


def validar_paciente(datos):
    errores = []

    for variable, (minimo, maximo) in RANGOS_DISENO.items():
        valor = datos.get(variable)

        if valor is None:
            errores.append(f"Falta ingresar: {variable}.")
            continue

        try:
            valor = float(valor)
        except (ValueError, TypeError):
            errores.append(f"El valor de {variable} no es numérico.")
            continue

        if not minimo <= valor <= maximo:
            errores.append(
                f"{variable}: debe estar entre {minimo} y {maximo}."
            )

    # Comprobación de coherencia de la lectura
    sistolica = datos.get("presion_sistolica")
    diastolica = datos.get("presion_diastolica")

    if sistolica is not None and diastolica is not None:
        try:
            if float(sistolica) <= float(diastolica):
                errores.append(
                    "La presión sistólica debe ser mayor que la diastólica."
                )
        except (ValueError, TypeError):
            pass

    return len(errores) == 0, errores
