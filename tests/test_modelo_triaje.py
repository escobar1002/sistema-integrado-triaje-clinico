
# ============================================================
# PRUEBAS UNITARIAS DEL MODELO MATEMATICO DE TRIAJE
# Sistema Integrado de Soporte para Triaje Clinico
# Proyecto de Ingenieria I - Etapa 4
# ============================================================

import unittest

from modules.modelo_triaje import (
    COEFICIENTES,
    validar_variables,
    calcular_indice_prioridad,
    clasificar_indice_prioridad,
    evaluar_modelo_triaje
)


class TestModeloTriaje(unittest.TestCase):

    def setUp(self):
        """
        Datos normalizados de un paciente simulado.
        Todas las variables están orientadas al riesgo.
        """

        self.variables = {
            "spo2": 0.20,
            "pa": 0.50,
            "frecuencia_cardiaca": 0.30,
            "frecuencia_respiratoria": 0.40,
            "temperatura": 0.25,
            "dolor": 0.50,
            "edad": 0.50
        }

    # --------------------------------------------------------
    # 1. PRUEBAS DE COEFICIENTES
    # --------------------------------------------------------

    def test_suma_coeficientes(self):
        self.assertAlmostEqual(
            sum(COEFICIENTES.values()), 1.0
        )

    def test_coeficiente_edad(self):
        self.assertEqual(COEFICIENTES["edad"], 0.05)

    # --------------------------------------------------------
    # 2. PRUEBAS DE VALIDACION
    # --------------------------------------------------------

    def test_validacion_variables_validas(self):
        validar_variables(self.variables)

    def test_variable_faltante(self):
        variables = self.variables.copy()
        del variables["edad"]

        with self.assertRaises(ValueError):
            validar_variables(variables)

    def test_variable_fuera_de_rango(self):
        variables = self.variables.copy()
        variables["spo2"] = 1.5

        with self.assertRaises(ValueError):
            calcular_indice_prioridad(variables)

    def test_variable_negativa(self):
        variables = self.variables.copy()
        variables["edad"] = -0.1

        with self.assertRaises(ValueError):
            calcular_indice_prioridad(variables)

    def test_variable_nula(self):
        variables = self.variables.copy()
        variables["edad"] = None

        with self.assertRaises(ValueError):
            calcular_indice_prioridad(variables)

    def test_variable_no_numerica(self):
        variables = self.variables.copy()
        variables["edad"] = "adulto"

        with self.assertRaises(ValueError):
            calcular_indice_prioridad(variables)

    # --------------------------------------------------------
    # 3. PRUEBAS DEL CALCULO DEL INDICE
    # --------------------------------------------------------

    def test_calculo_indice_prioridad(self):

        resultado = calcular_indice_prioridad(
            self.variables
        )

        self.assertIn("indice_prioridad", resultado)
        self.assertIn("contribuciones", resultado)
        self.assertIn("coeficientes", resultado)

        self.assertEqual(
            resultado["indice_prioridad"], 0.355
        )

    def test_contribucion_edad(self):

        resultado = calcular_indice_prioridad(
            self.variables
        )

        self.assertEqual(
            resultado["contribuciones"]["edad"], 0.025
        )

    def test_edad_sin_aporte(self):

        variables = self.variables.copy()
        variables["edad"] = 0.0

        resultado = calcular_indice_prioridad(variables)

        self.assertEqual(
            resultado["contribuciones"]["edad"], 0.0
        )

    def test_aporte_maximo_edad(self):

        variables = self.variables.copy()
        variables["edad"] = 1.0

        resultado = calcular_indice_prioridad(variables)

        self.assertEqual(
            resultado["contribuciones"]["edad"], 0.05
        )

    # --------------------------------------------------------
    # 4. PRUEBAS DE CLASIFICACION EXPERIMENTAL
    # --------------------------------------------------------

    def test_clasificacion_nivel_I(self):
        self.assertEqual(
            clasificar_indice_prioridad(0.85),
            "Nivel I"
        )

    def test_clasificacion_nivel_II(self):
        self.assertEqual(
            clasificar_indice_prioridad(0.70),
            "Nivel II"
        )

    def test_clasificacion_nivel_III(self):
        self.assertEqual(
            clasificar_indice_prioridad(0.50),
            "Nivel III"
        )

    def test_clasificacion_nivel_IV(self):
        self.assertEqual(
            clasificar_indice_prioridad(0.30),
            "Nivel IV"
        )

    def test_clasificacion_nivel_V(self):
        self.assertEqual(
            clasificar_indice_prioridad(0.10),
            "Nivel V"
        )

    def test_limites_clasificacion(self):

        self.assertEqual(
            clasificar_indice_prioridad(0.80),
            "Nivel I"
        )

        self.assertEqual(
            clasificar_indice_prioridad(0.60),
            "Nivel II"
        )

        self.assertEqual(
            clasificar_indice_prioridad(0.40),
            "Nivel III"
        )

        self.assertEqual(
            clasificar_indice_prioridad(0.20),
            "Nivel IV"
        )

        self.assertEqual(
            clasificar_indice_prioridad(0.00),
            "Nivel V"
        )

    def test_indice_fuera_de_rango(self):

        with self.assertRaises(ValueError):
            clasificar_indice_prioridad(1.5)

        with self.assertRaises(ValueError):
            clasificar_indice_prioridad(-0.1)

    # --------------------------------------------------------
    # 5. PRUEBA DE EVALUACION COMPLETA
    # --------------------------------------------------------

    def test_evaluacion_completa(self):

        resultado = evaluar_modelo_triaje(
            self.variables
        )

        self.assertIn("indice_prioridad", resultado)
        self.assertIn("nivel_experimental", resultado)
        self.assertEqual(
            resultado["nivel_experimental"],
            "Nivel IV"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
