
def calcular_pam(presion_sistolica, presion_diastolica):
    """
    Calcula la presión arterial media (PAM)
    a partir de la presión sistólica y diastólica.

    Fórmula:
    PAM = (PAS + 2 * PAD) / 3
    """

    if presion_sistolica is None or presion_diastolica is None:
        raise ValueError("La presión sistólica y diastólica son obligatorias.")

    if not isinstance(presion_sistolica, (int, float)):
        raise ValueError("La presión sistólica debe ser numérica.")

    if not isinstance(presion_diastolica, (int, float)):
        raise ValueError("La presión diastólica debe ser numérica.")

    if presion_sistolica <= 0 or presion_diastolica <= 0:
        raise ValueError("Los valores de presión deben ser mayores que cero.")

    if presion_sistolica <= presion_diastolica:
        raise ValueError("La presión sistólica debe ser mayor que la diastólica.")

    pam = (presion_sistolica + 2 * presion_diastolica) / 3

    return round(pam, 2)
