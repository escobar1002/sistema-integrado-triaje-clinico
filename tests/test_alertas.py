import unittest
from modules.alertas import detectar_condiciones_criticas


class TestAlertas(unittest.TestCase):

    def test_sin_alertas(self):
        datos = {
            "spo2": 98,
            "temperatura": 36.5
        }

        resultado = detectar_condiciones_criticas(datos)

        self.assertEqual(resultado["cantidad_alertas"], 0)

    def test_dos_alertas(self):
        datos = {
            "spo2": 80,
            "temperatura": 43
        }

        resultado = detectar_condiciones_criticas(datos)

        self.assertEqual(resultado["cantidad_alertas"], 2)

    def test_alerta_solo_saturacion(self):
        datos = {
            "spo2": 80,
            "temperatura": 36.5
        }

        resultado = detectar_condiciones_criticas(datos)

        self.assertEqual(resultado["cantidad_alertas"], 1)


if __name__ == "__main__":
    unittest.main()