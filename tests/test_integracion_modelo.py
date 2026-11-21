
import unittest

from modules.normalizacion import normalizar_paciente
from modules.modelo_triaje import evaluar_modelo_triaje


class TestIntegracionModelo(unittest.TestCase):

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

    def test_integracion_normalizacion_modelo(self):

        normalizados = normalizar_paciente(self.paciente)

        variables_modelo = {
            "spo2": normalizados["spo2_riesgo"],
            "pa": normalizados["pam_riesgo"],
            "frecuencia_cardiaca":
                normalizados["frecuencia_cardiaca_riesgo"],
            "frecuencia_respiratoria":
                normalizados["frecuencia_respiratoria_riesgo"],
            "temperatura": normalizados["temperatura_riesgo"],
            "dolor": normalizados["dolor"],
            "edad": normalizados["edad_riesgo"]
        }

        resultado = evaluar_modelo_triaje(variables_modelo)

        self.assertIn("indice_prioridad", resultado)
        self.assertIn("nivel_experimental", resultado)
        self.assertIn("contribuciones", resultado)

        self.assertGreaterEqual(
            resultado["indice_prioridad"], 0
        )

        self.assertLessEqual(
            resultado["indice_prioridad"], 1
        )

    def test_resultado_paciente_ejemplo(self):

        normalizados = normalizar_paciente(self.paciente)

        variables_modelo = {
            "spo2": normalizados["spo2_riesgo"],
            "pa": normalizados["pam_riesgo"],
            "frecuencia_cardiaca":
                normalizados["frecuencia_cardiaca_riesgo"],
            "frecuencia_respiratoria":
                normalizados["frecuencia_respiratoria_riesgo"],
            "temperatura": normalizados["temperatura_riesgo"],
            "dolor": normalizados["dolor"],
            "edad": normalizados["edad_riesgo"]
        }

        resultado = evaluar_modelo_triaje(variables_modelo)

        self.assertEqual(resultado["indice_prioridad"], 0.0983)
        self.assertEqual(resultado["nivel_experimental"], "Nivel V")


if __name__ == "__main__":
    unittest.main()
