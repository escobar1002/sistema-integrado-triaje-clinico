
import csv
import shutil
from pathlib import Path

CARPETA_DATOS = Path(__file__).parent / "data"
ARCHIVO = CARPETA_DATOS / "registros_simulados.csv"
RESPALDO = CARPETA_DATOS / "registros_simulados_respaldo.csv"

COLUMNAS_ADICIONALES = [
    "indice_prioridad",
    "nivel_experimental",
    "cantidad_alertas"
]


def reparar_historial():

    if not ARCHIVO.exists():
        print("No existe el archivo de historial.")
        return

    # Crear respaldo antes de modificar
    shutil.copy2(ARCHIVO, RESPALDO)

    # Leer todas las filas originales
    with open(ARCHIVO, "r", encoding="utf-8-sig", newline="") as archivo:
        lector = csv.reader(archivo)
        filas = list(lector)

    if not filas:
        print("El archivo está vacío.")
        return

    encabezado = filas[0]
    registros = filas[1:]

    # Agregar las columnas nuevas si todavía no existen
    for columna in COLUMNAS_ADICIONALES:
        if columna not in encabezado:
            encabezado.append(columna)

    cantidad_columnas = len(encabezado)

    # Verificar que ninguna fila tenga más columnas
    for numero, fila in enumerate(registros, start=2):
        if len(fila) > cantidad_columnas:
            raise ValueError(
                f"La fila {numero} tiene {len(fila)} columnas. "
                f"Se esperaban máximo {cantidad_columnas}. "
                "No se modificó el archivo original."
            )

    # Completar registros antiguos con campos vacíos
    filas_reparadas = []

    for fila in registros:
        fila = fila + [""] * (cantidad_columnas - len(fila))
        filas_reparadas.append(fila)

    # Escribir el historial reparado
    with open(ARCHIVO, "w", encoding="utf-8-sig", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(encabezado)
        escritor.writerows(filas_reparadas)

    print("Historial reparado correctamente.")
    print(f"Registros conservados: {len(filas_reparadas)}")
    print(f"Columnas finales: {cantidad_columnas}")
    print(f"Respaldo creado en: {RESPALDO}")


if __name__ == "__main__":
    reparar_historial()
