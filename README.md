# Machine Learning Supervisado: Regresión Logística y Algoritmos de Clasificación

> **Universidad de Cundinamarca** — Especialización en Inteligencia Artificial
> **CADI – Introducción a Machine Learning — Periodo 2026-2**
> Docente: Ing. Wilson Rojas Reales, PhD(c)
> Estudiante: **Yohann Javier Nikolay Salazar Cardenas**

Repositorio del informe práctico sobre **aprendizaje supervisado**: regresión logística y
los clasificadores **Decision Tree, Random Forest, SVM, KNN y Naive Bayes**, aplicados a
dos casos de estudio empresariales (defectos de producción y rotación de empleados) y a un
experimento puente con la práctica anterior de regresión lineal.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yohann0011/ml-supervisado-clasificacion-ucundinamarca/blob/main/notebooks/01_caso_produccion.ipynb)

## Contenido

| Caso | Problema | Mejor modelo (F1) | Notebook |
|---|---|---|---|
| 1. Producción (XYZ Inc.) | Lote defectuoso (0/1) con temperatura, presión, velocidad, horas y vibración | SVM RBF (0.588) + umbral costo-óptimo | `notebooks/01_caso_produccion.ipynb` |
| 2. Recursos humanos | Renuncia (0/1) con edad, salario, permanencia, satisfacción y horas extras | Random Forest (0.699) | `notebooks/02_caso_rrhh.ipynb` |
| 3. Puente inmobiliario | OLS de la práctica anterior validado + segmento premium (0/1) | Regresión logística (0.947) | `notebooks/03_puente_inmobiliario.ipynb` |

Todos los notebooks están **ejecutados de extremo a extremo** (salidas, tablas y figuras
embebidas) y son reproducibles en Google Colab con un solo clic: el primer celda clona este
repositorio y fija la semilla 42.

## Estructura

```
├── README.md
├── requirements.txt
├── notebooks/            # 3 notebooks ejecutados en Colab
├── data/                 # datasets (CSV), métricas (CSV/JSON) y modelos (joblib)
├── figuras/              # 32 figuras del informe (PNG 200 dpi)
├── src/                  # generador de datos, pipeline experimental, puente, notebooks
├── scripts_semana_anterior/   # regresión lineal múltiple (práctica previa) + Excel
└── informe/              # Informe_ML_Supervisado.pdf y .docx (APA 7)
```

## Reproducción local

```bash
pip install -r requirements.txt
python src/generar_datos.py        # regenera los datasets (semilla 42)
python src/analisis.py             # experimento completo de los casos 1 y 2
python src/puente_inmobiliario.py  # validación OLS + clasificación puente
cd ../informe && python3 build_final.py   # reconstruye PDF y DOCX del informe
```

## Datos

Los conjuntos de `data/` son **sintéticos y reproducibles** (mecanismo generador logístico
con semilla fija, calibrado a prevalencias realistas de 17.7 % y 21.9 %), construidos porque
la guía publica los scripts de referencia pero no los datos. Las relaciones entre variables
siguen el enunciado del caso de estudio. Para usar datos reales basta reemplazar los CSV.

## Métricas y evaluación

Accuracy, balanced accuracy, precision, recall, F1-Score, especificidad, AUC-ROC,
Average Precision, MCC, kappa, matriz de confusión, validación cruzada 5-fold,
GridSearchCV, curvas de aprendizaje, de calibración, de umbral y análisis de costos
(COP 2 500 000 por falso negativo y COP 180 000 por falso positivo en producción).

## Informe

El informe completo en formato PDF y Word (normas APA 7.ª edición, con citas
bibliográficas) está en `informe/`. Incluye la resolución de los 17 interrogantes de la
guía para el caso de rotación de empleados.
