"""
make_notebooks.py
=================
Construye y EJECUTA los tres notebooks del repositorio (con salidas y figuras
embebidas) para que cualquier revisor pueda verificarlos en GitHub o Colab.
"""

from __future__ import annotations

import nbformat as nbf
from nbclient import NotebookClient
from pathlib import Path

RAIZ = Path("/home/user/ml_informe")
NB = RAIZ / "notebooks"
NB.mkdir(exist_ok=True)

HEADER = """# Machine Learning Supervisado — Regresión Logística y Clasificación
**Universidad de Cundinamarca — Esp. Inteligencia Artificial**
**CADI – Introducción a Machine Learning — Periodo 2026-2**
Docente: Ing. Wilson Rojas Reales, PhD(c)

> Repositorio del proyecto: https://github.com/[su-usuario-de-github]/ml-supervisado-clasificacion-ucundinamarca
> Este notebook se ejecuta de extremo a extremo en **Google Colab** (runtime CPU).
"""

SETUP = """try:
    import google.colab  # noqa
    EN_COLAB = True
    !git clone --depth 1 https://github.com/[su-usuario-de-github]/ml-supervisado-clasificacion-ucundinamarca repo 2>/dev/null
    import os; os.chdir('repo')
except Exception:
    EN_COLAB = False

import sys
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
    n.cells = []
    for tipo, txt in celdas:
        n.cells.append(nbf.v4.new_markdown_cell(txt) if tipo == "md"
                       else nbf.v4.new_code_cell(txt))
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    return n


def ejecutar(n, nombre):
    dest = NB / nombre
    client = NotebookClient(n, timeout=1500, kernel_name="python3",
                            resources={"metadata": {"path": str(RAIZ)}})
    client.execute()
    nbf.write(n, dest)
    print("OK", dest)


# ============================================================ 01 PRODUCCIÓN
c1 = [
    ("md", HEADER + """
# Notebook 01 — Caso de estudio: Predicción de defectos en producción (XYZ Inc.)

**Objetivo:** predecir si un lote de producción será defectuoso (1) o no (0) a partir de
temperatura, presión, velocidad, horas de operación y vibración, comparando seis
clasificadores supervisados: Regresión Logística, Árbol de Decisión, Random Forest, SVM,
KNN y Naive Bayes, con métricas de precisión, recall, F1, especificidad, AUC-ROC y matriz
de confusión, más un análisis de umbral y de costos.
"""),
    ("code", SETUP),
    ("md", "## 1. Carga de datos y análisis exploratorio"),
    ("code", """df = pd.read_csv(DIR_DATA / 'caso_produccion.csv')
print(df.shape)
df.describe().T.round(2)"""),
    ("code", """print('Distribución de clases:')
print(df.defectuoso.value_counts(normalize=True).round(4))
sns.countplot(x=df.defectuoso, palette=['#4C9F70', '#D1495B'])
plt.title('Desbalance de clases (17.7 % defectuosos)')
plt.show()"""),
    ("code", """from analisis import ResultadoCaso
res = ResultadoCaso('Producción', 'defectuoso', {0: 'No defectuoso', 1: 'Defectuoso'})
res.df = df
res.features = ['temperatura', 'presion', 'velocidad', 'horas_operacion', 'vibracion']"""),
    ("md", "## 2. Entrenamiento y evaluación de los seis clasificadores\n"
           "Partición estratificada 80/20 (semilla 42); estandarización dentro de Pipeline; "
           "`class_weight='balanced'` en los modelos con función de pérdida."),
    ("code", """from analisis import ejecutar_caso
res = ejecutar_caso('Producción', df, 'defectuoso',
                    {0: 'No defectuoso', 1: 'Defectuoso'}, 'prod')"""),
    ("code", """tabla = res.tabla_metricas()
tabla.round(3)"""),
    ("md", "## 3. Curvas ROC y Precision-Recall, matrices de confusión"),
    ("code", """for f in sorted(DIR_FIGS.glob('prod_*.png')):
    display(Image(filename=f))"""),
    ("md", "## 4. Reportes de clasificación completos (precision, recall, F1 por clase)"),
    ("code", """for nombre, m in res.metricas.items():
    print('=' * 70); print(nombre); print('=' * 70)
    print(m['reporte'])
    print(f\"TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}\")"""),
    ("md", "## 5. Umbral de decisión y análisis de costos (COP 2.5 M por FN, COP 180 k por FP)"),
    ("code", """umb = res.extra['umbrales']
umb"""),
    ("code", """import pandas as _pd
_pd.DataFrame(res.extra['costos']).T"""),
    ("md", "## 6. Ajuste de hiperparámetros (GridSearchCV, 5 pliegues, scoring=F1)"),
    ("code", """_pd.DataFrame(res.extra['grid_search']).T"""),
    ("md", """## 7. Interpretación
* El **SVM (RBF)** logra el mejor F1 (0.588) y el menor costo esperado: captura la no
  linealidad del efecto térmico que los modelos lineales no representan.
* **KNN** tiene especificidad altísima pero recall 0.283: inaceptable cuando el falso
  negativo cuesta COP 2 500 000.
* Bajar el umbral del SVM a 0.15 lleva el recall a 0.811 y el costo de COP 73.2 M a
  COP 34.0 M por cada 300 lotes de prueba: el umbral es una decisión de negocio.
* La vibración y la temperatura dominan la importancia permutacional: son las palancas
  de calidad de la planta.
"""),
]

# ============================================================ 02 RRHH
c2 = [
    ("md", HEADER + """
# Notebook 02 — Caso de estudio: Rotación de empleados

**Objetivo:** predecir la renuncia (1) o permanencia (0) de los empleados a partir de edad,
salario (log10), años de permanencia, satisfacción y horas extras; comparar los seis
clasificadores y experimentar el efecto de añadir clima laboral y modalidad de trabajo
(pregunta 6 de la guía).
"""),
    ("code", SETUP),
    ("code", """df = pd.read_csv(DIR_DATA / 'caso_rrhh.csv')
df.describe().T.round(2)"""),
    ("code", """print('Distribución de clases:')
print(df.renuncia.value_counts(normalize=True).round(4))"""),
    ("code", """from analisis import ResultadoCaso, graficar_eda, ejecutar_caso
base = ['edad', 'salario_log10', 'anos_permanencia', 'satisfaccion', 'horas_extras']
res = ejecutar_caso('Rotación de empleados', df, 'renuncia',
                    {0: 'Permanece', 1: 'Renuncia'}, 'rrhh',
                    features=base,
                    features_extra=['clima_laboral', 'modalidad_trabajo'])"""),
    ("code", "res.tabla_metricas().round(3)"),
    ("code", """for f in sorted(DIR_FIGS.glob('rrhh_*.png')):
    display(Image(filename=f))"""),
    ("md", "## Experimento pregunta 6: ¿mejora el modelo con clima_laboral y modalidad_trabajo?"),
    ("code", """p6 = res.extra['pregunta6']
comp = pd.DataFrame({
    'F1 base': {k: v['f1'] for k, v in p6['Modelo base (5 variables)'].items()},
    'F1 ampliado': {k: v['f1'] for k, v in p6['Modelo ampliado (+2 variables)'].items()},
    'AUC ampliado': {k: v['roc_auc'] for k, v in p6['Modelo ampliado (+2 variables)'].items()},
})
comp['delta'] = (comp['F1 ampliado'] - comp['F1 base']).round(3)
comp.round(3)"""),
    ("code", "display(Image(filename=DIR_FIGS / 'rrhh_14_pregunta6.png'))"),
    ("code", """for nombre, m in res.metricas.items():
    print('=' * 70); print(nombre); print('=' * 70)
    print(m['reporte'])"""),
    ("md", """## Interpretación
* **Random Forest** obtiene el mejor F1 (0.699) y el mejor MCC: equilibra detección de
  renuncias (recall 0.679) con alarmas útiles (precision 0.720).
* **Regresión logística** es el mejor rankeador (AUC 0.884) y el más interpretable: cada
  punto de satisfacción multiplica la razón de momios de renuncia por 0.21.
* La transformación **log10(salario)** es crítica: sin ella, Naive Bayes degenera a F1 0.000.
* Añadir clima laboral y modalidad mejora el AUC (mejor ranking) aunque el F1 con umbral
  0.50 exige recalibración del punto de corte.
"""),
]

# ============================================================ 03 PUENTE
c3 = [
    ("md", HEADER + """
# Notebook 03 — Puente con la práctica anterior: regresión lineal → clasificación

Valida en Python el modelo OLS de la semana anterior (Excel `=ESTIMACION.LINEAL`, hoja
*Datos_y_Modelo*) y convierte el mismo fenómeno inmobiliario en un problema de
clasificación binaria (segmento premium), resolviéndolo con regresión logística y los
cinco clasificadores de la guía.
"""),
    ("code", SETUP),
    ("code", """from puente_inmobiliario import parte1_validacion_ols, parte2_clasificacion
ols = parte1_validacion_ols()
pd.DataFrame({'excel': ols['excel'], 'python': ols['python']}).T"""),
    ("code", "pd.DataFrame(ols['tabla'])"),
    ("code", """cls = parte2_clasificacion()
pd.DataFrame(cls['metricas']).T.round(3)"""),
    ("code", "print(cls['logistica']['reporte'])"),
    ("md", """## Lectura
* Los coeficientes OLS de Python coinciden con los del Excel hasta 1e-7: mismos
  cálculos, dos motores distintos.
* Binarizado el objetivo, la regresión logística alcanza F1 0.947 y sus odds ratios
  estandarizados (área 24.3) muestran la misma relación lineal del modelo continuo,
  ahora en escala de momios.
* Regresión responde «¿cuánto?»; clasificación responde «¿cuál clase?» y se evalúa con
  métricas de error por clase, no con R².
"""),
]

if __name__ == "__main__":
    for celdas, nombre in ((c1, "01_caso_produccion.ipynb"),
                           (c2, "02_caso_rrhh.ipynb"),
                           (c3, "03_puente_inmobiliario.ipynb")):
        ejecutar(nb(celdas), nombre)
