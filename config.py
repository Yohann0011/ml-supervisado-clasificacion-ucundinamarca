"""
config.py
=========
Parámetros globales del proyecto "Machine Learning Supervisado: Regresión
Logística y Algoritmos de Clasificación".

Universidad de Cundinamarca - Esp. Inteligencia Artificial
CADI - Introducción a Machine Learning - Periodo 2026-2

NOTA PARA EL ESTUDIANTE:
    Modifique ÚNICAMENTE las variables de la sección "IDENTIFICACIÓN" y el
    GITHUB_REPO. Todo el informe (PDF y DOCX) y el README se regeneran a
    partir de este archivo.
"""

from pathlib import Path

# ============================================================
# 1. IDENTIFICACIÓN (completar con sus datos)
# ============================================================
NOMBRE_ESTUDIANTE = "Yohann Javier Nikolay Salazar Cardenas"
CODIGO_ESTUDIANTE = ""  # el estudiante indicó que no se incluye código
PROGRAMA = "Especialización en Inteligencia Artificial"
ASIGNATURA = "CADI - Introducción a Machine Learning"
PERIODO = "Periodo 2026-2"
DOCENTE = "Ing. Wilson Rojas Reales, PhD(c)"
INSTITUCION = "Universidad de Cundinamarca"
CIUDAD = "Fusagasugá - Cundinamarca, Colombia"
FECHA_ENTREGA = "8 de octubre de 2026"

# ============================================================
# 2. REPOSITORIO GITHUB  <-- REEMPLAZAR POR SU URL REAL
# ============================================================
GITHUB_USUARIO = "Yohann0011"
GITHUB_REPO_NOMBRE = "ml-supervisado-clasificacion-ucundinamarca"
GITHUB_REPO = f"https://github.com/{GITHUB_USUARIO}/{GITHUB_REPO_NOMBRE}"
GITHUB_ES_PLACEHOLDER = GITHUB_USUARIO.startswith("[")

# ============================================================
# 3. RUTAS DEL PROYECTO
# ============================================================
RAIZ = Path(__file__).parent  # relativa: robusta frente a vistas del sandbox
DIR_DATA = RAIZ / "data"
DIR_FIGS = RAIZ / "figuras"
DIR_SRC = RAIZ / "src"
DIR_REPORT = RAIZ.parent / "informe"
DIR_NOTEBOOKS = RAIZ / "notebooks"

for d in (DIR_DATA, DIR_FIGS, DIR_SRC, DIR_REPORT, DIR_NOTEBOOKS):
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# 4. PARÁMETROS EXPERIMENTALES (reproducibilidad)
# ============================================================
SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
N_CASO_PRODUCCION = 1500   # lotes de producción
N_CASO_RRHH = 1200         # empleados

CLASES_PRODUCCION = {0: "No defectuoso", 1: "Defectuoso"}
CLASES_RRHH = {0: "Permanece", 1: "Renuncia"}

# Costos de negocio (para el análisis costo-sensitivo del caso Producción)
COSTO_FALSO_NEGATIVO = 2_500_000   # COP: lote defectuoso que llega al cliente
COSTO_FALSO_POSITIVO = 180_000     # COP: inspección/reproceso de un lote sano
MONEDA = "COP"

# ============================================================
# 5. ESTILOS VISUALES
# ============================================================
PALETA_MODELOS = {
    "Regresión Logística": "#1f77b4",
    "Árbol de Decisión": "#ff7f0e",
    "Random Forest": "#2ca02c",
    "SVM": "#d62728",
    "KNN": "#9467bd",
    "Naive Bayes": "#8c564b",
}
