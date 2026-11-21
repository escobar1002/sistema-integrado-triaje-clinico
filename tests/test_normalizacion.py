
import unittest

from modules.normalizacion import (
    normalizar_variable,
    transformar_edad_riesgo,
    calcular_riesgo_frecuencia_cardiaca,
    calcular_riesgo_frecuencia_respiratoria,
    calcular_riesgo_temperatura,
    calcular_riesgo_pam,
    normalizar_pam,
    normalizar_paciente
)


class TestNormalizarVariable(unittest.TestCase):

    def test_normalizacion_correcta(self):
        self.assertEqual(normalizar_variable(50, 0, 100), 0.5)

    def test_valor_minimo(self):
        self.assertEqual(normalizar_variable(0, 0, 100), 0.0)

    def test_valor_maximo(self):
        self.assertEqual(normalizar_variable(100, 0, 100), 1.0)

    def test_valor_fuera_de_rango(self):
        with self.assertRaises(ValueError):
            normalizar_variable(110, 0, 100)

    def test_valor_none(self):
        with self.assertRaises(ValueError):
            normalizar_variable(None, 0, 100)

    def test_limites_invalidos(self):
        with self.assertRaises(ValueError):
            normalizar_variable(50, 100, 0)


class TestTransformacionEdad(unittest.TestCase):

    def test_menor_de_65(self):
        self.assertEqual(transformar_edad_riesgo(40), 0.0)

    def test_desde_65(self):
        self.assertEqual(transformar_edad_riesgo(65), 0.5)

    def test_desde_80(self):
        self.assertEqual(transformar_edad_riesgo(80), 1.0)

    def test_edad_cero(self):
        self.assertEqual(transformar_edad_riesgo(0), 0.0)

    def test_edad_maxima(self):
        self.assertEqual(transformar_edad_riesgo(120), 1.0)

    def test_edad_fuera_de_rango(self):
        with self.assertRaises(ValueError):
            transformar_edad_riesgo(121)

    def test_edad_none(self):
        with self.assertRaises(ValueError):
            transformar_edad_riesgo(None)


class TestRiesgoFrecuenciaCardiaca(unittest.TestCase):

    def test_limites_de_frecuencia_cardiaca(self):
        casos = [
            (40, 1.0),
            (41, 0.25),
            (50, 0.25),
            (51, 0.0),
            (90, 0.0),
            (91, 0.25),
            (110, 0.25),
            (111, 0.50),
            (130, 0.50),
            (131, 1.0)
        ]

        for valor, esperado in casos:
            with self.subTest(valor=valor):
                self.assertEqual(
                    calcular_riesgo_frecuencia_cardiaca(valor),
                    esperado
                )

    def test_frecuencia_fuera_de_rango(self):
        with self.assertRaises(ValueError):
            calcular_riesgo_frecuencia_cardiaca(221)


class TestRiesgoFrecuenciaRespiratoria(unittest.TestCase):

    def test_limites_de_frecuencia_respiratoria(self):
        casos = [
            (8, 1.0),
            (9, 0.25),
            (11, 0.25),
            (12, 0.0),
            (20, 0.0),
            (21, 0.50),
            (24, 0.50),
            (25, 1.0)
        ]

        for valor, esperado in casos:
            with self.subTest(valor=valor):
                self.assertEqual(
                    calcular_riesgo_frecuencia_respiratoria(valor),
                    esperado
                )

    def test_frecuencia_fuera_de_rango(self):
        with self.assertRaises(ValueError):
            calcular_riesgo_frecuencia_respiratoria(51)


class TestRiesgoTemperatura(unittest.TestCase):

    def test_limites_de_temperatura(self):
        casos = [
            (35, 1.0),
            (35.1, 0.25),
            (36, 0.25),
            (36.1, 0.0),
            (38, 0.0),
            (38.1, 0.25),
            (39, 0.25),
            (39.1, 0.50),
            (41, 0.50),
            (41.1, 1.0)
        ]

        for valor, esperado in casos:
            with self.subTest(valor=valor):
                self.assertEqual(
                    calcular_riesgo_temperatura(valor),
                    esperado
                )

    def test_temperatura_fuera_de_rango(self):
        with self.assertRaises(ValueError):
            calcular_riesgo_temperatura(43)


class TestRiesgoPAM(unittest.TestCase):

    def test_limites_de_riesgo_pam(self):
        casos = [
            (59, 1.0),
            (60, 0.50),
            (64, 0.50),
            (65, 0.0),
            (100, 0.0),
            (101, 0.25)
        ]

        for valor, esperado in casos:
            with self.subTest(valor=valor):
                self.assertEqual(calcular_riesgo_pam(valor), esperado)

    def test_pam_negativa(self):
        with self.assertRaises(ValueError):
            calcular_riesgo_pam(-1)

    def test_normalizacion_pam(self):
        self.assertEqual(normalizar_pam(80), 0.5)

    def test_pam_baja_limitada(self):
        self.assertEqual(normalizar_pam(40), 0)

    def test_pam_alta_limitada(self):
        self.assertEqual(normalizar_pam(120), 1)


class TestNormalizarPaciente(unittest.TestCase):

    def setUp(self):
        self.paciente = {
            "edad": 70,
            "dolor": 4,
            "frecuencia_cardiaca": 80,
            "frecuencia_respiratoria": 18,
            "temperatura": 37,
            "spo2": 98,
            "presion_sistolica": 120,
            "presion_diastolica": 80
        }

    def test_campos_normalizados(self):
        resultado = normalizar_paciente(self.paciente)

        self.assertIn("edad", resultado)
        self.assertIn("edad_riesgo", resultado)
        self.assertIn("dolor", resultado)
        self.assertIn("frecuencia_cardiaca_riesgo", resultado)
        self.assertIn("frecuencia_respiratoria_riesgo", resultado)
        self.assertIn("temperatura_riesgo", resultado)
        self.assertIn("spo2_riesgo", resultado)
        self.assertIn("pam", resultado)
        self.assertIn("pam_riesgo", resultado)

    def test_valores_esperados(self):
        resultado = normalizar_paciente(self.paciente)

        self.assertEqual(resultado["edad"], 0.5833)
        self.assertEqual(resultado["edad_riesgo"], 0.5)
        self.assertEqual(resultado["dolor"], 0.4)
        self.assertEqual(resultado["frecuencia_cardiaca_riesgo"], 0.0)
        self.assertEqual(resultado["frecuencia_respiratoria_riesgo"], 0.0)
        self.assertEqual(resultado["temperatura_riesgo"], 0.0)
        self.assertEqual(resultado["spo2_riesgo"], 0.1333)
        self.assertEqual(resultado["pam"], 93.33)
        self.assertEqual(resultado["pam_riesgo"], 0.0)

    def test_paciente_sin_presion_arterial(self):
        paciente = {
            "edad": 30,
            "dolor": 2,
            "frecuencia_cardiaca": 80,
            "frecuencia_respiratoria": 18,
            "temperatura": 37,
            "spo2": 98
        }

        resultado = normalizar_paciente(paciente)

        self.assertNotIn("pam", resultado)
        self.assertNotIn("pam_riesgo", resultado)


if __name__ == "__main__":
    unittest.main()
