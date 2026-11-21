import unittest
from modules.alertas import detectar_condiciones_criticas


class TestAlertaSaturacion(unittest.TestCase):

    def test_alerta_por_saturacion_baja(self):
        datos = {
            "spo2": 84,
            "temperatura": 37,
            "presion_sistolica": 120,
            "presion_diastolica": 80
        }

        resultado = detectar_condiciones_criticas(datos)

        self.assertGreater(resultado["cantidad_alertas"], 0)

        alertas_saturacion = [
            alerta for alerta in resultado["alertas"]
            if "saturacion" in alerta["variable"].lower()
        ]

        self.assertTrue(alertas_saturacion)
        self.assertEqual(alertas_saturacion[0]["nivel"], "ROJO")

    def test_saturacion_sin_alerta(self):
        datos = {
            "spo2": 98,
            "temperatura": 37,
            "presion_sistolica": 120,
            "presion_diastolica": 80
        }

        resultado = detectar_condiciones_criticas(datos)

        self.assertFalse(resultado["alertas"])


if __name__ == "__main__":
    unittest.main()