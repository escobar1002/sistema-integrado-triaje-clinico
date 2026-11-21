
import unittest
from modules.presion_arterial import calcular_pam


class TestPresionArterial(unittest.TestCase):

    def test_calculo_pam(self):
        resultado = calcular_pam(120, 80)
        self.assertEqual(resultado, 93.33)

    def test_presion_sistolica_menor_que_diastolica(self):
        with self.assertRaises(ValueError):
            calcular_pam(70, 90)

    def test_presiones_iguales(self):
        with self.assertRaises(ValueError):
            calcular_pam(80, 80)

    def test_presion_negativa(self):
        with self.assertRaises(ValueError):
            calcular_pam(-120, 80)

    def test_valor_no_numerico(self):
        with self.assertRaises(ValueError):
            calcular_pam("120", 80)


if __name__ == "__main__":
    unittest.main()
