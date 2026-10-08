"""Construye y ejecuta los 3 notebooks del repositorio (rutas relativas)."""
import os
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

RAIZ = Path(os.environ.get("RAIZ_ML", "."))
NB = RAIZ / "notebooks"
os.makedirs(str(NB), exist_ok=True)

HEADER = """# Machine Learning Supervisado — Regresión Logística y Clasificación
**Universidad de Cundinamarca — Esp. Inteligencia Artificial**
**CADI – Introducción a Machine Learning — Periodo 2026-2**
Docente: Ing. Wilson Rojas Reales, PhD(c)

> Repositorio: https://github.com/[su-usuario-de-github]/ml-supervisado-clasificacion-ucundinamarca
> Este notebook se ejecuta de extremo a extremo en **Google Colab** (runtime CPU).
"""

SETUP = """try:
    import google.colab  # noqa
    EN_COLAB = True
    !git clone --depth 1 https://github.com/[su-usuario-de-github]/ml-supervisado-clasificacion-ucundinamarca repo 2>/dev/null
    import os
    if os.path.isdir('repo'): os.chdir('repo')
except Exception:
    EN_COLAB = False

import os, sys
from pathlib import Path
RAIZ = Path(os.environ.get('RAIZ_ML', '.'))
for p in (str(RAIZ), str(RAIZ / 'src')):
    if p not in sys.path:
        sys.path.insert(0, p)

%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from IPython.display import Image, display
from config import SEED, DIR_DATA, DIR_FIGS
print('Semilla:', SEED, '| datos en:', DIR_DATA)
"""

def nb(celdas):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(t) if k == "md" else nbf.v4.new_code_cell(t)
               for k, t in celdas]
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    return n

def ejecutar(n, nombre):
    dest = NB / nombre
    client = NotebookClient(n, timeout=1500, kernel_name="python3",
                            resources={"metadata": {"path": str(RAIZ)}})
    client.execute()
    nbf.write(n, str(dest))
    print("OK", dest)

c1 = [
    ("md", HEADER + """
# Notebook 01 — Predicción de defectos en producción (XYZ Inc.)

**Objetivo:** predecir si un lote será defectuoso (1) o no (0) con temperatura, presión,
velocidad, horas de operación y vibración; comparar Regresión Logística, Árbol de Decisión,
Random Forest, SVM, KNN y Naive Bayes con precisión, recall, F1, especificidad, AUC-ROC,
matriz de confusión, umbral de decisión y análisis de costos.
"""),
    ("code", SETUP),
    ("md", "## 1. Carga de datos y análisis exploratorio"),
    ("code", "df = pd.read_csv(DIR_DATA / 'caso_produccion.csv')\ndf.describe().T.round(2)"),
    ("code", "print(df.defectuoso.value_counts(normalize=True).round(4))\nsns.countplot(x=df.defectuoso, palette=['#4C9F70','#D1495B'])\nplt.title('Clases: 17.7 % defectuosos')\nplt.show()"),
    ("md", "## 2. Entrenamiento y evaluación (partición 80/20 estratificada, semilla 42)"),
    ("code", "from analisis import ejecutar_caso\nres = ejecutar_caso('Producción', df, 'defectuoso', {0: 'No defectuoso', 1: 'Defectuoso'}, 'prod')"),
    ("code", "res.tabla_metricas().round(3)"),
    ("md", "## 3. Figuras del experimento (ROC, PR, confusión, importancia, umbrales, costos)"),
    ("code", "for f in sorted(DIR_FIGS.glob('prod_*.png')):\n    display(Image(filename=str(f)))"),
    ("md", "## 4. Reportes de clasificación por modelo"),
    ("code", "for nombre, m in res.metricas.items():\n    print('='*70); print(nombre); print('='*70); print(m['reporte'])"),
    ("md", "## 5. Umbral de decisión y costo económico"),
    ("code", "pd.DataFrame(res.extra['umbrales'])"),
    ("code", "pd.DataFrame(res.extra['costos']).T"),
    ("md", "## 6. GridSearchCV (5 pliegues, scoring=F1)"),
    ("code", "pd.DataFrame(res.extra['grid_search']).T"),
    ("md", """## 7. Interpretación
* **SVM (RBF)**: mejor F1 (0.588) y menor costo; captura la no linealidad térmica.
* **KNN**: recall 0.283 → inaceptable cuando el falso negativo cuesta COP 2.5 M.
* Umbral 0.15 en SVM: recall 0.811 y costo de COP 34.0 M vs 73.2 M del umbral 0.50.
* Vibración y temperatura dominan la importancia permutacional.
"""),
]

c2 = [
    ("md", HEADER + """
# Notebook 02 — Rotación de empleados

**Objetivo:** predecir renuncia (1) / permanencia (0) con edad, salario (log10),
permanencia, satisfacción y horas extras; comparar los seis clasificadores y evaluar el
aporte de clima laboral y modalidad de trabajo (pregunta 6 de la guía).
"""),
    ("code", SETUP),
    ("code", "df = pd.read_csv(DIR_DATA / 'caso_rrhh.csv')\ndf.describe().T.round(2)"),
    ("code", "print(df.renuncia.value_counts(normalize=True).round(4))"),
    ("code", "from analisis import ejecutar_caso\nbase = ['edad','salario_log10','anos_permanencia','satisfaccion','horas_extras']\nres = ejecutar_caso('Rotación de empleados', df, 'renuncia', {0:'Permanece',1:'Renuncia'}, 'rrhh', features=base, features_extra=['clima_laboral','modalidad_trabajo'])"),
    ("code", "res.tabla_metricas().round(3)"),
    ("code", "for f in sorted(DIR_FIGS.glob('rrhh_*.png')):\n    display(Image(filename=str(f)))"),
    ("md", "## Experimento pregunta 6"),
    ("code", "p6 = res.extra['pregunta6']\ncomp = pd.DataFrame({'F1 base': {k:v['f1'] for k,v in p6['Modelo base (5 variables)'].items()}, 'F1 ampliado': {k:v['f1'] for k,v in p6['Modelo ampliado (+2 variables)'].items()}, 'AUC ampliado': {k:v['roc_auc'] for k,v in p6['Modelo ampliado (+2 variables)'].items()}})\ncomp['delta'] = (comp['F1 ampliado']-comp['F1 base']).round(3)\ncomp.round(3)"),
    ("code", "for nombre, m in res.metricas.items():\n    print('='*70); print(nombre); print('='*70); print(m['reporte'])"),
    ("md", """## Interpretación
* **Random Forest**: mejor F1 (0.699) y MCC (0.617).
* **Regresión logística**: mejor AUC (0.884) y odds ratios interpretables.
* **log10(salario)** evita que Naive Bayes degenere (F1 0.000 → 0.581).
* Clima laboral y modalidad mejoran el AUC: hay señal, pero exige recalibrar el umbral.
"""),
]

c3 = [
    ("md", HEADER + """
# Notebook 03 — Puente con la práctica anterior (regresión lineal → clasificación)

Valida el OLS inmobiliario de la semana anterior (Excel `=ESTIMACION.LINEAL`) y resuelve el
mismo fenómeno como clasificación binaria (segmento premium).
"""),
    ("code", SETUP),
    ("code", "from puente_inmobiliario import parte1_validacion_ols, parte2_clasificacion\nols = parte1_validacion_ols()\npd.DataFrame({'excel': ols['excel'], 'python': ols['python']}).T"),
    ("code", "pd.DataFrame(ols['tabla'])"),
    ("code", "cls = parte2_clasificacion()\npd.DataFrame(cls['metricas']).T.round(3)"),
    ("code", "print(cls['logistica']['reporte'])"),
    ("code", "for f in sorted(DIR_FIGS.glob('puente_*.png')):\n    display(Image(filename=str(f)))"),
    ("md", """## Lectura
* Coeficientes OLS idénticos al Excel hasta 1e-7.
* Binarizado, la regresión logística logra F1 0.947; odds ratio del área estandarizada: 24.3.
* Regresión responde «¿cuánto?» (R², RMSE); clasificación «¿cuál clase?» (recall, F1, AUC).
"""),
]

if __name__ == "__main__":
    for celdas, nombre in ((c1, "01_caso_produccion.ipynb"),
                           (c2, "02_caso_rrhh.ipynb"),
                           (c3, "03_puente_inmobiliario.ipynb")):
        ejecutar(nb(celdas), nombre)
