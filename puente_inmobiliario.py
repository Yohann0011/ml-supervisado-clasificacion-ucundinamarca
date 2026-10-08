"""
puente_inmobiliario.py
======================
Sección puente del informe: conecta la práctica de la semana anterior
(regresión lineal múltiple inmobiliaria) con la práctica actual
(clasificación supervisada).

Parte 1 - Validación: se replican en Python (scikit-learn) los coeficientes
          OLS que el Excel de la semana pasada calculaba con =ESTIMACION.LINEAL
          sobre las 12 viviendas de la hoja "Datos_y_Modelo".
Parte 2 - Puente conceptual: se toma el generador sintético del script
          RegresionLogistica_Vivienda.py (100 viviendas) y se binariza el
          objetivo (segmento premium = precio sobre la mediana) para resolver
          el MISMO problema con regresión logística y los cinco clasificadores
          de la guía, comparando el paradigma de regresión con el de
          clasificación.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "src"))
from config import DIR_DATA, DIR_FIGS, SEED  # noqa: E402
from analisis import construir_modelos, metricas_completas, PALETA_MODELOS  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})

# Valores publicados por el Excel de la semana pasada (hoja Datos_y_Modelo)
EXCEL = {
    "coef_habitaciones": 12214.188000781713,
    "coef_area": 2466.6308383818641,
    "intercepto": 38466.875122141864,
    "r2": 0.99974998800395198,
}


def parte1_validacion_ols() -> dict:
    """Replica en Python el modelo OLS del Excel (12 viviendas)."""
    df = pd.DataFrame({
        "Area_m2": [50, 65, 70, 80, 95, 110, 120, 135, 150, 165, 180, 200],
        "Habitaciones": [1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
        "Precio_USD": [175000, 220000, 235000, 275000, 310000, 345000,
                       385000, 420000, 460000, 505000, 540000, 595000],
    })
    X = df[["Area_m2", "Habitaciones"]]
    y = df["Precio_USD"]
    mdl = LinearRegression().fit(X, y)
    y_pred = mdl.predict(X)
    out = {
        "n": int(len(df)),
        "python": {
            "intercepto": round(float(mdl.intercept_), 6),
            "coef_area": round(float(mdl.coef_[0]), 6),
            "coef_habitaciones": round(float(mdl.coef_[1]), 6),
            "r2": round(float(r2_score(y, y_pred)), 9),
            "rmse": round(float(np.sqrt(mean_squared_error(y, y_pred))), 2),
            "mae": round(float(mean_absolute_error(y, y_pred)), 2),
        },
        "excel": EXCEL,
        "tabla": df.assign(Predicho=y_pred.round(2), Error=(y - y_pred).round(2))
                   .to_dict(orient="records"),
    }
    diffs = {
        "intercepto": abs(out["python"]["intercepto"] - EXCEL["intercepto"]),
        "coef_area": abs(out["python"]["coef_area"] - EXCEL["coef_area"]),
        "coef_habitaciones": abs(out["python"]["coef_habitaciones"] - EXCEL["coef_habitaciones"]),
        "r2": abs(out["python"]["r2"] - EXCEL["r2"]),
    }
    out["diferencias_absolutas"] = {k: round(v, 9) for k, v in diffs.items()}
    print("OLS replicado:", out["python"])
    print("Diferencias vs Excel:", out["diferencias_absolutas"])

    # Figura: reales vs predichos + residuos (réplica del gráfico de la semana pasada)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    axes[0].scatter(y, y_pred, color="#1f77b4", edgecolor="k", s=60, zorder=3)
    axes[0].plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2, label="Ajuste perfecto")
    axes[0].set_xlabel("Precio real (USD)")
    axes[0].set_ylabel("Precio predicho (USD)")
    axes[0].set_title(f"OLS semana anterior replicado (R² = {out['python']['r2']:.5f})")
    axes[0].legend(fontsize=9)
    resid = y - y_pred
    axes[1].axhline(0, color="r", ls="--", lw=2)
    axes[1].scatter(y_pred, resid, color="#9467bd", edgecolor="k", s=60, zorder=3)
    axes[1].set_xlabel("Precio predicho (USD)")
    axes[1].set_ylabel("Residuo (real − predicho)")
    axes[1].set_title(f"Residuos (RMSE = USD {out['python']['rmse']:,.0f})")
    fig.suptitle("Validación del modelo de regresión lineal múltiple de la práctica anterior",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(DIR_FIGS / "puente_01_ols.png")
    plt.close(fig)
    return out


def parte2_clasificacion() -> dict:
    """Binariza el problema inmobiliario y lo resuelve con los 6 clasificadores."""
    # Mismo generador del script RegresionLogistica_Vivienda.py (semilla 42)
    np.random.seed(42)
    n = 100
    area_m2 = np.random.uniform(40, 200, n)
    habitaciones = np.random.randint(1, 6, n)
    precio = 30000 + 2500 * area_m2 + 15000 * habitaciones + np.random.normal(0, 15000, n)
    df = pd.DataFrame({"Area_m2": area_m2, "Habitaciones": habitaciones, "Precio": precio})

    # Binarización: segmento premium = precio por encima de la mediana muestral
    umbral = float(df["Precio"].median())
    df["premium"] = (df["Precio"] > umbral).astype(int)
    X = df[["Area_m2", "Habitaciones"]]
    y = df["premium"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    out = {"n": int(n), "umbral_premium": round(umbral, 2),
           "tasa_premium": round(float(y.mean()), 4), "metricas": {}}

    fig, ax = plt.subplots(figsize=(6.4, 5.5))
    for nombre, modelo in construir_modelos().items():
        modelo.fit(Xtr, ytr)
        yp = modelo.predict(Xte)
        pr = modelo.predict_proba(Xte)[:, 1]
        m = metricas_completas(yte, yp, pr)
        cv = cross_val_score(modelo, Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                             scoring="f1").mean()
        m["cv_f1"] = round(float(cv), 4)
        tn, fp, fn, tp = m["tn"], m["fp"], m["fn"], m["tp"]
        out["metricas"][nombre] = m
        fpr, tpr, _ = roc_curve(yte, pr)
        ax.plot(fpr, tpr, lw=2, color=PALETA_MODELOS.get(nombre),
                label=f"{nombre} (AUC = {m['roc_auc']:.3f})")
        print(f"{nombre}: Acc={m['accuracy']:.3f} Prec={m['precision']:.3f} "
              f"Rec={m['recall']:.3f} F1={m['f1']:.3f} CM[TN={tn} FP={fp} FN={fn} TP={tp}]")

    # Regresión logística interpretable (coeficientes sobre variables estandarizadas)
    lr = construir_modelos()["Regresión Logística"]
    lr.fit(Xtr, ytr)
    coefs = lr.named_steps["clf"].coef_[0]
    out["logistica"] = {
        "coef_area_est": round(float(coefs[0]), 4),
        "coef_hab_est": round(float(coefs[1]), 4),
        "odds_area": round(float(np.exp(coefs[0])), 4),
        "odds_hab": round(float(np.exp(coefs[1])), 4),
        "reporte": classification_report(yte, lr.predict(Xte),
                                          target_names=["Estándar", "Premium"], digits=3),
    }
    print("Logística estandarizada:", out["logistica"])

    ax.plot([0, 1], [0, 1], "k--", lw=1.1, label="Azar")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Clasificación del segmento premium (puente con la semana anterior)")
    ax.legend(fontsize=8.2, loc="lower right")
    fig.savefig(DIR_FIGS / "puente_02_roc.png")
    plt.close(fig)
    return out


if __name__ == "__main__":
    res = {"ols": parte1_validacion_ols(), "clasificacion": parte2_clasificacion()}
    (DIR_DATA / "puente_resultados.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nGuardado data/puente_resultados.json")
