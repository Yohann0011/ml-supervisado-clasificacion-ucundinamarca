"""
analisis.py
===========
Núcleo experimental del informe: entrena y evalúa Regresión Logística y los
cinco algoritmos de clasificación de la guía (Árbol de Decisión, Random
Forest, SVM, KNN y Naive Bayes) sobre los dos casos de estudio, y genera
todas las métricas y figuras del informe.

Basado en la API de scikit-learn (Pedregosa et al., 2011).
"""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    COSTO_FALSO_NEGATIVO,
    COSTO_FALSO_POSITIVO,
    CV_FOLDS,
    DIR_DATA,
    DIR_FIGS,
    MONEDA,
    PALETA_MODELOS,
    SEED,
    TEST_SIZE,
)

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
sns.set_palette("deep")


# ============================================================================
# ESTRUCTURAS
# ============================================================================
@dataclass
class ResultadoCaso:
    """Contenedor de todos los resultados de un caso de estudio."""

    nombre: str
    target: str
    nombres_clases: dict
    features: list = field(default_factory=list)
    df: pd.DataFrame = None
    X_train: pd.DataFrame = None
    X_test: pd.DataFrame = None
    y_train: np.ndarray = None
    y_test: np.ndarray = None
    modelos: dict = field(default_factory=dict)
    metricas: dict = field(default_factory=dict)
    y_prob: dict = field(default_factory=dict)
    y_pred: dict = field(default_factory=dict)
    cv: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    # ---------------- utilidades ----------------
    def tabla_metricas(self) -> pd.DataFrame:
        filas = []
        for m, d in self.metricas.items():
            filas.append(
                {
                    "Modelo": m,
                    "Accuracy": d["accuracy"],
                    "Bal. Acc.": d["balanced_accuracy"],
                    "Precision": d["precision"],
                    "Recall": d["recall"],
                    "F1-Score": d["f1"],
                    "Especificidad": d["specificity"],
                    "AUC-ROC": d["roc_auc"],
                    "AP (PR-AUC)": d["average_precision"],
                    "Acc. train": d["train_accuracy"],
                    "Acc. CV-5": d["cv_f1_mean"],
                    "σ CV": d["cv_f1_std"],
                }
            )
        return pd.DataFrame(filas).sort_values("F1-Score", ascending=False).reset_index(drop=True)

    def matriz_confusion(self, modelo: str) -> np.ndarray:
        return confusion_matrix(self.y_test, self.y_pred[modelo])


# ============================================================================
# 1. MODELOS
# ============================================================================
def construir_modelos() -> dict:
    """Devuelve los 6 clasificadores supervisados dentro de un Pipeline.

    El `StandardScaler` se aplica SOLO a los modelos sensibles a la escala
    (Regresión Logística, SVM, KNN). Los basados en árboles y Naive Bayes son
    invariantes a transformaciones monótonas de escala, pero se dejan en el
    mismo Pipeline para garantizar condiciones idénticas de comparación.
    """
    return {
        "Regresión Logística": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED)),
            ]
        ),
        "Árbol de Decisión": Pipeline(
            [("clf", DecisionTreeClassifier(class_weight="balanced", random_state=SEED))]
        ),
        "Random Forest": Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300, max_depth=None, class_weight="balanced",
                        random_state=SEED, n_jobs=-1,
                    ),
                )
            ]
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED)),
            ]
        ),
        "KNN": Pipeline(
            [("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=9, weights="distance"))]
        ),
        "Naive Bayes": Pipeline([("clf", GaussianNB())]),
    }


def metricas_completas(y_true, y_pred, y_prob) -> dict:
    """Calcula el set completo de métricas de clasificación binaria."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "specificity": round(tn / (tn + fp), 4) if (tn + fp) else 0.0,
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4) if y_prob is not None else np.nan,
        "average_precision": round(average_precision_score(y_true, y_prob), 4) if y_prob is not None else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fpr": round(fp / (fp + tn), 4) if (fp + tn) else 0.0,
        "fnr": round(fn / (fn + tp), 4) if (fn + tp) else 0.0,
        "mcc": round(_mcc(tp, tn, fp, fn), 4),
        "kappa": round(_kappa(y_true, y_pred), 4),
    }


def _mcc(tp, tn, fp, fn) -> float:
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float(num / den) if den else 0.0


def _kappa(y_true, y_pred) -> float:
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(y_true, y_pred))


# ============================================================================
# 2. EXPERIMENTO
# ============================================================================
def ejecutar_caso(
    nombre: str,
    df: pd.DataFrame,
    target: str,
    nombres_clases: dict,
    prefijo: str,
    features: list | None = None,
    features_extra: list | None = None,
) -> ResultadoCaso:
    """Ejecuta el pipeline completo (EDA -> train/test -> 6 modelos -> métricas)."""
    res = ResultadoCaso(nombre=nombre, target=target, nombres_clases=nombres_clases)
    res.df = df

    if features is None:
        features = [c for c in df.columns
                    if c != target and not c.endswith("_id") and c not in ("lote",)]
    res.features = list(features)
    X, y = df[features], df[target].values

    print(f"\n{'#' * 74}\n# CASO: {nombre}\n# n = {len(df)} | variables = {len(features)} | "
          f"clase positiva = {y.mean():.2%}\n{'#' * 74}")

    # --- Partición estratificada 80/20 -------------------------------------
    res.X_train, res.X_test, res.y_train, res.y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    print(f"Train: {res.X_train.shape} (positivos {res.y_train.mean():.2%}) | "
          f"Test: {res.X_test.shape} (positivos {res.y_test.mean():.2%})")

    # --- Línea base: clasificador mayoritario ------------------------------
    mayor = np.bincount(res.y_train).argmax()
    res.extra["baseline_accuracy"] = round((res.y_test == mayor).mean(), 4)
    print(f"Línea base (clase mayoritaria = {mayor}): accuracy = {res.extra['baseline_accuracy']:.4f}")

    # --- Entrenamiento y evaluación de los 6 modelos -----------------------
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    for nombre_modelo, modelo in construir_modelos().items():
        modelo.fit(res.X_train, res.y_train)
        y_pred = modelo.predict(res.X_test)
        try:
            y_prob = modelo.predict_proba(res.X_test)[:, 1]
        except AttributeError:
            y_prob = modelo.decision_function(res.X_test)
        cv_f1 = cross_val_score(modelo, res.X_train, res.y_train, cv=cv, scoring="f1", n_jobs=-1)

        m = metricas_completas(res.y_test, y_pred, y_prob)
        m["train_accuracy"] = round(accuracy_score(res.y_train, modelo.predict(res.X_train)), 4)
        m["train_f1"] = round(f1_score(res.y_train, modelo.predict(res.X_train), zero_division=0), 4)
        m["cv_f1_mean"] = round(cv_f1.mean(), 4)
        m["cv_f1_std"] = round(cv_f1.std(), 4)
        m["gap_overfitting"] = round(m["train_accuracy"] - m["accuracy"], 4)
        m["reporte"] = classification_report(
            res.y_test, y_pred, target_names=[nombres_clases[0], nombres_clases[1]],
            digits=3, zero_division=0,
        )
        res.metricas[nombre_modelo] = m
        res.modelos[nombre_modelo] = modelo
        res.y_pred[nombre_modelo] = y_pred
        res.y_prob[nombre_modelo] = y_prob

        print(f"\n>>> {nombre_modelo}")
        print(f"    Acc={m['accuracy']:.3f} Prec={m['precision']:.3f} Rec={m['recall']:.3f} "
              f"F1={m['f1']:.3f} AUC={m['roc_auc']:.3f} AP={m['average_precision']:.3f} "
              f"Esp={m['specificity']:.3f} MCC={m['mcc']:.3f}")
        print(f"    CM [TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}] | "
              f"CV-F1={m['cv_f1_mean']:.3f}±{m['cv_f1_std']:.3f} | "
              f"gap overfitting={m['gap_overfitting']:+.3f}")

    # --- Ajuste fino del mejor modelo por F1 --------------------------------
    mejor = max(res.metricas, key=lambda k: res.metricas[k]["f1"])
    res.extra["mejor_modelo_f1"] = mejor
    _grid_search(res, mejor)

    # --- Importancia de variables (permutacional) ---------------------------
    res.extra["importancia"] = _importancia_permutacional(res)

    # --- Análisis de umbral de decisión -------------------------------------
    res.extra["umbrales"] = _analisis_umbral(res, mejor)

    # --- Análisis de costo (solo caso producción, pero genérico) -------------
    res.extra["costos"] = _analisis_costo(res)

    # --- Experimento pregunta 6: añadir 2 variables -------------------------
    if features_extra:
        res.extra["pregunta6"] = _experimento_variables_extra(df, features, features_extra, target, nombres_clases)

    return res


def _grid_search(res: ResultadoCaso, nombre_modelo: str) -> dict:
    """Búsqueda de hiperparámetros (5-fold CV) sobre los modelos clave."""
    espacios = {
        "Regresión Logística": {"clf__C": [0.01, 0.1, 1, 10, 100]},
        "Árbol de Decisión": {"clf__max_depth": [3, 5, 8, 12, None],
                              "clf__min_samples_leaf": [1, 5, 10, 20]},
        "Random Forest": {"clf__n_estimators": [100, 300, 500],
                          "clf__max_depth": [6, 10, None],
                          "clf__min_samples_leaf": [1, 3, 5]},
        "SVM": {"clf__C": [0.1, 1, 10, 50], "clf__gamma": ["scale", 0.01, 0.1, 1]},
        "KNN": {"clf__n_neighbors": [3, 5, 7, 9, 15, 21, 31]},
        "Naive Bayes": {"clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]},
    }
    salida = {}
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    for nom, espacio in espacios.items():
        gs = GridSearchCV(
            construir_modelos()[nom], espacio, scoring="f1", cv=cv, n_jobs=-1, refit=True
        )
        gs.fit(res.X_train, res.y_train)
        y_pred = gs.predict(res.X_test)
        try:
            y_prob = gs.predict_proba(res.X_test)[:, 1]
        except AttributeError:
            y_prob = None
        m = metricas_completas(res.y_test, y_pred, y_prob)
        salida[nom] = {
            "mejores_params": {k: (int(v) if isinstance(v, (np.integer,)) else v)
                               for k, v in gs.best_params_.items()},
            "mejor_f1_cv": round(gs.best_score_, 4),
            "test_accuracy": m["accuracy"], "test_precision": m["precision"],
            "test_recall": m["recall"], "test_f1": m["f1"],
            "test_roc_auc": m["roc_auc"], "test_ap": m["average_precision"],
            "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"],
        }
        print(f"    [GridSearch] {nom}: {salida[nom]['mejores_params']} -> "
              f"F1_cv={salida[nom]['mejor_f1_cv']:.3f} | F1_test={m['f1']:.3f} "
              f"Rec={m['recall']:.3f} AUC={m['roc_auc']:.3f}")
    res.extra["grid_search"] = salida


def _importancia_permutacional(res: ResultadoCaso) -> dict:
    """Importancia por permutación sobre el conjunto de prueba (modelo base)."""
    salida = {}
    for nom in ["Regresión Logística", "Random Forest"]:
        modelo = res.modelos[nom]
        r = permutation_importance(
            modelo, res.X_test, res.y_test, n_repeats=20, random_state=SEED,
            scoring="f1", n_jobs=-1,
        )
        salida[nom] = {
            f: {"importancia": round(float(i), 4), "std": round(float(s), 4)}
            for f, i, s in zip(res.X_test.columns, r.importances_mean, r.importances_std)
        }
    # Coeficientes estandarizados de la Regresión Logística (odds ratio)
    lr = res.modelos["Regresión Logística"]
    coefs = lr.named_steps["clf"].coef_[0]
    salida["coeficientes_logisticos"] = {
        f: {"coeficiente": round(float(c), 4), "odds_ratio": round(float(np.exp(c)), 4)}
        for f, c in zip(res.X_test.columns, coefs)
    }
    salida["intercept_logistico"] = round(float(lr.named_steps["clf"].intercept_[0]), 4)
    return salida


def _analisis_umbral(res: ResultadoCaso, modelo: str) -> pd.DataFrame:
    """Recall/Precision/F1/costo para distintos umbrales de decisión."""
    prob = res.y_prob[modelo]
    filas = []
    for t in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]:
        yp = (prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(res.y_test, yp).ravel()
        filas.append(
            {
                "umbral": t,
                "precision": round(precision_score(res.y_test, yp, zero_division=0), 4),
                "recall": round(recall_score(res.y_test, yp, zero_division=0), 4),
                "f1": round(f1_score(res.y_test, yp, zero_division=0), 4),
                "accuracy": round(accuracy_score(res.y_test, yp), 4),
                "fp": int(fp), "fn": int(fn), "tp": int(tp), "tn": int(tn),
                "costo_total": int(fp * COSTO_FALSO_POSITIVO + fn * COSTO_FALSO_NEGATIVO),
            }
        )
    df = pd.DataFrame(filas)
    return df


def _analisis_costo(res: ResultadoCaso) -> dict:
    """Costo esperado por modelo usando los costos de negocio definidos."""
    salida = {}
    for nom, m in res.metricas.items():
        costo = m["fp"] * COSTO_FALSO_POSITIVO + m["fn"] * COSTO_FALSO_NEGATIVO
        salida[nom] = {
            "costo_total": int(costo),
            "costo_por_lote": round(costo / len(res.y_test), 1),
            "costo_ahorrado_vs_baseline": None,
        }
    # Costo de NO usar ningún modelo (todos pasan sin inspección)
    n_positivos = int(res.y_test.sum())
    costo_no_modelo = n_positivos * COSTO_FALSO_NEGATIVO
    for nom in salida:
        salida[nom]["costo_ahorrado_vs_baseline"] = int(costo_no_modelo - salida[nom]["costo_total"])
    salida["_referencia"] = {
        "costo_sin_modelo": int(costo_no_modelo),
        "costo_fn": COSTO_FALSO_NEGATIVO,
        "costo_fp": COSTO_FALSO_POSITIVO,
        "moneda": MONEDA,
        "positivos_en_test": n_positivos,
    }
    return salida


def _experimento_variables_extra(df, features_base, features_extra, target, nombres_clases) -> dict:
    """Compara el desempeño con y sin las 2 variables adicionales (pregunta 6)."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    conjuntos = {
        "Modelo base (5 variables)": features_base,
        "Modelo ampliado (+2 variables)": features_base + features_extra,
    }
    salida = {}
    for etiqueta, cols in conjuntos.items():
        X = df[cols]
        y = df[target].values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
        parcial = {}
        for nom, modelo in construir_modelos().items():
            modelo.fit(Xtr, ytr)
            yp = modelo.predict(Xte)
            pr = modelo.predict_proba(Xte)[:, 1]
            cvf1 = cross_val_score(modelo, Xtr, ytr, cv=cv, scoring="f1", n_jobs=-1).mean()
            m = metricas_completas(yte, yp, pr)
            parcial[nom] = {"accuracy": m["accuracy"], "precision": m["precision"],
                            "recall": m["recall"], "f1": m["f1"], "roc_auc": m["roc_auc"],
                            "cv_f1": round(float(cvf1), 4)}
        salida[etiqueta] = parcial
        mejor = max(parcial, key=lambda k: parcial[k]["f1"])
        print(f"    [Pregunta 6] {etiqueta}: mejor={mejor} F1={parcial[mejor]['f1']:.3f} "
              f"AUC={parcial[mejor]['roc_auc']:.3f}")
    base = salida["Modelo base (5 variables)"]
    ampl = salida["Modelo ampliado (+2 variables)"]
    delta = {k: round(ampl[k]["f1"] - base[k]["f1"], 4) for k in base}
    salida["delta_f1"] = delta
    salida["delta_f1_promedio"] = round(float(np.mean(list(delta.values()))), 4)
    return salida


# ============================================================================
# 3. FIGURAS
# ============================================================================
def _fig(nombre: str) -> Path:
    return DIR_FIGS / nombre


def graficar_eda(res: ResultadoCaso, prefijo: str) -> None:
    """Figuras de análisis exploratorio de datos."""
    df, target = res.df, res.target
    num = [c for c in res.features if df[c].dtype != object]

    # 3.1 Matriz de correlación ------------------------------------------------
    corr = df[num + [target]].corr()
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.6, cbar_kws={"shrink": 0.85}, ax=ax)
    ax.set_title(f"Matriz de correlaciones - Caso {res.nombre}")
    fig.savefig(_fig(f"{prefijo}_01_correlaciones.png"))
    plt.close(fig)
    corr.to_csv(DIR_DATA / f"{prefijo}_correlaciones.csv")

    # 3.2 Distribución de la variable objetivo --------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    vc = df[target].value_counts().sort_index()
    colores = ["#4C9F70", "#D1495B"]
    ejes = axes[0].bar([res.nombres_clases[int(k)] for k in vc.index], vc.values, color=colores, width=0.55)
    axes[0].bar_label(ejes, [f"{v}\n({v/len(df):.1%})" for v in vc.values], fontsize=9, padding=3)
    axes[0].set_title("Distribución de clases (desbalance)")
    axes[0].set_ylabel("Frecuencia")
    axes[0].set_ylim(0, vc.max() * 1.22)
    plt.close(fig)

    # 3.2b Densidad de cada variable por clase (paneles individuales) --------
    k = len(num)
    cols = 3
    rows = int(np.ceil((k + 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 2.5 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i, f in enumerate(num):
        sns.kdeplot(data=df, x=f, hue=target, ax=axes[i], common_norm=False,
                    fill=True, alpha=0.35, linewidth=1.7, palette=colores,
                    hue_order=sorted(df[target].unique()), legend=(i == 0))
        axes[i].set_title(f, fontsize=10)
        axes[i].set_ylabel("")
        axes[i].tick_params(labelsize=8)
        if axes[i].get_legend():
            handles = axes[i].get_legend().legend_handles
            axes[i].legend(handles,
                           [res.nombres_clases[int(c)] for c in sorted(df[target].unique())],
                           title=res.target, fontsize=8, title_fontsize=8.5)
    # Panel libre: diagrama de dispersión de las 2 variables más importantes
    imp = res.extra.get("importancia", {}).get("Random Forest", {})
    top2 = sorted(imp, key=lambda x: -imp[x]["importancia"])[:2] if imp else num[:2]
    ax = axes[k]
    sns.scatterplot(data=df.sample(600, random_state=SEED), x=top2[0], y=top2[1],
                    hue=target, ax=ax, palette=colores, s=16, alpha=0.65,
                    hue_order=sorted(df[target].unique()), legend=False)
    ax.set_title(f"{top2[0]} vs {top2[1]} (muestra de 600)", fontsize=10)
    ax.tick_params(labelsize=8)
    for j in range(k + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Densidad de cada predictor según la clase - Caso {res.nombre}",
                 y=1.0, fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(_fig(f"{prefijo}_02_distribucion.png"))
    plt.close(fig)

    # 3.3 Boxplots por clase ---------------------------------------------------
    k = len(num)
    cols = 3
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.05 * cols, 2.55 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i, f in enumerate(num):
        sns.boxplot(data=df, x=target, y=f, ax=axes[i], palette=colores, width=0.5,
                    hue=target, legend=False, fliersize=2.2)
        axes[i].set_xticklabels([res.nombres_clases[int(t)] for t in sorted(df[target].unique())],
                                fontsize=8)
        axes[i].set_title(f, fontsize=10)
        axes[i].set_xlabel("")
    for j in range(k, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Caso {res.nombre}: distribución de cada variable según la clase",
                 y=1.005, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(_fig(f"{prefijo}_03_boxplots.png"))
    plt.close(fig)


def graficar_modelos(res: ResultadoCaso, prefijo: str) -> None:
    """Curvas ROC, Precision-Recall, matrices de confusión e importancia."""
    y_test = res.y_test
    c0, c1 = res.nombres_clases[0], res.nombres_clases[1]

    # 3.4 Curvas ROC -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.3, 5.6))
    for nom in res.modelos:
        fpr, tpr, _ = roc_curve(y_test, res.y_prob[nom])
        ax.plot(fpr, tpr, lw=2.1, color=PALETA_MODELOS.get(nom),
                label=f"{nom} (AUC = {res.metricas[nom]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.1, label="Azar (AUC = 0.500)")
    ax.set_xlabel("Tasa de Falsos Positivos (1 - Especificidad)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (Recall)")
    ax.set_title(f"Curvas ROC - Caso {res.nombre}")
    ax.legend(loc="lower right", fontsize=8.6, framealpha=0.95)
    fig.savefig(_fig(f"{prefijo}_04_roc.png"))
    plt.close(fig)

    # 3.5 Curvas Precision-Recall ---------------------------------------------
    fig, ax = plt.subplots(figsize=(6.3, 5.6))
    base = y_test.mean()
    for nom in res.modelos:
        p, r, _ = precision_recall_curve(y_test, res.y_prob[nom])
        ax.plot(r, p, lw=2.1, color=PALETA_MODELOS.get(nom),
                label=f"{nom} (AP = {res.metricas[nom]['average_precision']:.3f})")
    ax.axhline(base, color="gray", ls="--", lw=1.1, label=f"Prevalencia ({base:.3f})")
    ax.set_xlabel("Recall (Sensibilidad)")
    ax.set_ylabel("Precision (VPP)")
    ax.set_title(f"Curvas Precision-Recall - Caso {res.nombre}")
    ax.legend(loc="lower left", fontsize=8.6, framealpha=0.95)
    fig.savefig(_fig(f"{prefijo}_05_precision_recall.png"))
    plt.close(fig)

    # 3.6 Matrices de confusión (2x3) -----------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 7.1))
    for ax, nom in zip(axes.ravel(), res.modelos):
        ConfusionMatrixDisplay.from_predictions(
            y_test, res.y_pred[nom], ax=ax, cmap="Blues", colorbar=False,
            display_labels=[c0, c1], values_format="d",
        )
        m = res.metricas[nom]
        ax.set_title(f"{nom}\nAcc={m['accuracy']:.3f}  Prec={m['precision']:.3f}  "
                     f"Rec={m['recall']:.3f}  F1={m['f1']:.3f}", fontsize=9.4)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Valor real")
        ax.tick_params(labelsize=8.5)
    fig.suptitle(f"Matrices de confusión (conjunto de prueba) - Caso {res.nombre}",
                 fontsize=12.5, fontweight="bold", y=1.002)
    fig.tight_layout()
    fig.savefig(_fig(f"{prefijo}_06_confusion.png"))
    plt.close(fig)

    # 3.7 Comparación de métricas ---------------------------------------------
    tabla = res.tabla_metricas()
    metrics_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    x = np.arange(len(tabla))
    w = 0.16
    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    for i, mt in enumerate(metrics_plot):
        ax.bar(x + (i - 2) * w, tabla[mt], w, label=mt)
    ax.set_xticks(x)
    ax.set_xticklabels(tabla["Modelo"], rotation=12, ha="right", fontsize=9)
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("Puntuación")
    ax.set_title(f"Comparación de métricas por modelo - Caso {res.nombre}")
    ax.legend(ncol=5, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    ax.axhline(res.extra["baseline_accuracy"], color="red", ls=":", lw=1.3)
    ax.text(len(tabla) - 0.45, res.extra["baseline_accuracy"] + 0.015,
            f"línea base (clase mayoritaria) = {res.extra['baseline_accuracy']:.3f}",
            color="red", fontsize=8, ha="right")
    for xi, v in zip(x, tabla["F1-Score"]):
        ax.text(xi + w, v + 0.015, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
    fig.tight_layout()
    fig.savefig(_fig(f"{prefijo}_07_comparacion_metricas.png"))
    plt.close(fig)

    # 3.8 Importancia de variables --------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3))
    for ax, nom in zip(axes, ["Regresión Logística", "Random Forest"]):
        imp = res.extra["importancia"][nom]
        s = pd.Series({k: v["importancia"] for k, v in imp.items()}).sort_values()
        e = [imp[k]["std"] for k in s.index]
        barras = ax.barh(range(len(s)), s.values, xerr=e, color="#1f77b4" if nom.startswith("Reg") else "#2ca02c",
                         alpha=0.85, error_kw={"lw": 0.9, "capsize": 2.5})
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels(s.index, fontsize=9)
        ax.set_xlabel("Caída del F1-Score al permutar la variable")
        ax.set_title(f"Importancia permutacional - {nom}", fontsize=10)
        ax.bar_label(barras, [f"{v:.3f}" for v in s.values], padding=3, fontsize=8)
        ax.set_xlim(0, max(0.05, s.max() * 1.28))
    fig.suptitle(f"Importancia de las variables predictoras - Caso {res.nombre}",
                 fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(_fig(f"{prefijo}_08_importancia.png"))
    plt.close(fig)

    # 3.9 Curvas de aprendizaje ------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.4))
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    for ax, nom in zip(axes.ravel(), res.modelos):
        sizes, tr, va = learning_curve(
            res.modelos[nom], res.X_train, res.y_train, cv=cv,
            train_sizes=np.linspace(0.15, 1.0, 7), scoring="f1", n_jobs=-1,
            random_state=SEED,
        )
        ax.plot(sizes, tr.mean(1), "o-", lw=1.7, ms=4, label="Train", color=PALETA_MODELOS.get(nom))
        ax.fill_between(sizes, tr.mean(1) - tr.std(1), tr.mean(1) + tr.std(1), alpha=0.15,
                        color=PALETA_MODELOS.get(nom))
        ax.plot(sizes, va.mean(1), "s--", lw=1.7, ms=4, label="Validación", color="#444444")
        ax.fill_between(sizes, va.mean(1) - va.std(1), va.mean(1) + va.std(1), alpha=0.13, color="#444444")
        ax.set_title(nom, fontsize=10)
        ax.set_ylim(0, 1.03)
        ax.set_xlabel("Tamaño del conjunto de entrenamiento", fontsize=8.5)
        ax.set_ylabel("F1-Score", fontsize=8.5)
        ax.tick_params(labelsize=8)
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Curvas de aprendizaje (diagnóstico de sesgo/varianza) - Caso {res.nombre}",
                 fontsize=12.5, fontweight="bold", y=1.005)
    fig.tight_layout()
    fig.savefig(_fig(f"{prefijo}_09_aprendizaje.png"))
    plt.close(fig)

    # 3.10 Análisis de umbral de decisión -------------------------------------
    umb = res.extra["umbrales"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3))
    for mt, c in zip(["precision", "recall", "f1"], ["#1f77b4", "#d62728", "#2ca02c"]):
        axes[0].plot(umb["umbral"], umb[mt], "o-", lw=1.9, ms=4.5, label=mt.capitalize(), color=c)
    axes[0].axvline(0.5, color="gray", ls="--", lw=1.1)
    axes[0].set_xlabel("Umbral de decisión")
    axes[0].set_ylabel("Puntuación")
    axes[0].set_title(f"Trade-off Precision/Recall - {res.extra['mejor_modelo_f1']}", fontsize=10)
    axes[0].legend(fontsize=8.5)
    mejor_f1 = umb.loc[umb["f1"].idxmax(), "umbral"]
    axes[0].axvline(mejor_f1, color="green", ls=":", lw=1.4)
    axes[0].text(mejor_f1 + 0.008, 0.15, f"umbral óptimo F1 = {mejor_f1:.2f}",
                 color="green", fontsize=8, rotation=90, va="bottom")

    w = 0.38
    xs = np.arange(len(umb))
    axes[1].bar(xs - w / 2, umb["tp"], w, label="TP", color="#2ca02c")
    axes[1].bar(xs + w / 2, umb["fp"], w, label="FP", color="#ff7f0e")
    axes[1].bar(xs - w / 2, umb["fn"], w, bottom=umb["tp"], label="FN", color="#d62728")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([f"{u:.2f}" for u in umb["umbral"]], fontsize=8.5)
    axes[1].set_xlabel("Umbral de decisión")
    axes[1].set_ylabel("Número de casos")
    axes[1].set_title("Composición de errores según el umbral", fontsize=10)
    axes[1].legend(fontsize=8.5, ncol=3)
    fig.suptitle(f"Análisis del umbral de decisión - Caso {res.nombre}",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(_fig(f"{prefijo}_10_umbrales.png"))
    plt.close(fig)

    # 3.11 Calibración de probabilidades --------------------------------------
    fig, ax = plt.subplots(figsize=(6.3, 5.3))
    for nom in ["Regresión Logística", "Random Forest", "SVM", "Naive Bayes"]:
        fx, my = calibration_curve(y_test, res.y_prob[nom], n_bins=8, strategy="quantile")
        ax.plot(fx, my, "o-", lw=1.8, ms=4.5, color=PALETA_MODELOS.get(nom), label=nom)
    ax.plot([0, 1], [0, 1], "k--", lw=1.1, label="Perfectamente calibrado")
    ax.set_xlabel("Probabilidad predicha")
    ax.set_ylabel("Frecuencia observada de la clase positiva")
    ax.set_title(f"Curvas de calibración - Caso {res.nombre}")
    ax.legend(fontsize=8.4, loc="upper left")
    fig.savefig(_fig(f"{prefijo}_11_calibracion.png"))
    plt.close(fig)


def graficar_arbol(res: ResultadoCaso, prefijo: str, max_depth: int = 3) -> None:
    """Árbol de decisión legible (profundidad limitada) + Random Forest."""
    fig, ax = plt.subplots(figsize=(15, 7.2))
    plot_tree(
        res.modelos["Árbol de Decisión"].named_steps["clf"],
        feature_names=res.features,
        class_names=[res.nombres_clases[0], res.nombres_clases[1]],
        filled=True, rounded=True, fontsize=8, max_depth=max_depth,
        impurity=True, proportion=False, ax=ax,
    )
    ax.set_title(f"Árbol de Decisión (primeros {max_depth} niveles) - Caso {res.nombre}", fontsize=12)
    fig.savefig(_fig(f"{prefijo}_12_arbol.png"))
    plt.close(fig)


def graficar_hiperparams(res: ResultadoCaso, prefijo: str) -> None:
    """Comparación del F1 antes/después del ajuste de hiperparámetros."""
    gs = res.extra["grid_search"]
    nombres = list(gs.keys())
    f1_base = [res.metricas[n]["f1"] for n in nombres]
    f1_tuned = [gs[n]["test_f1"] for n in nombres]
    x = np.arange(len(nombres))
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    ax.bar(x - 0.19, f1_base, 0.38, label="Configuración por defecto", color="#8fa8c8")
    ax.bar(x + 0.19, f1_tuned, 0.38, label="Después de GridSearchCV", color="#2ca02c")
    ax.bar_label(ax.containers[0], [f"{v:.3f}" for v in f1_base], padding=2, fontsize=8)
    ax.bar_label(ax.containers[1], [f"{v:.3f}" for v in f1_tuned], padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(nombres, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("F1-Score (conjunto de prueba)")
    ax.set_ylim(0, 1.08)
    ax.set_title(f"Efecto del ajuste de hiperparámetros (GridSearchCV, {CV_FOLDS}-fold) - Caso {res.nombre}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(_fig(f"{prefijo}_13_gridsearch.png"))
    plt.close(fig)


def graficar_pregunta6(res: ResultadoCaso, prefijo: str) -> None:
    """Impacto de añadir clima_laboral y modalidad_trabajo (pregunta 6)."""
    p6 = res.extra.get("pregunta6")
    if not p6:
        return
    modelos = list(p6["Modelo base (5 variables)"].keys())
    base = [p6["Modelo base (5 variables)"][m]["f1"] for m in modelos]
    ampl = [p6["Modelo ampliado (+2 variables)"][m]["f1"] for m in modelos]
    x = np.arange(len(modelos))
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    ax.bar(x - 0.19, base, 0.38, label="5 variables originales", color="#8fa8c8")
    ax.bar(x + 0.19, ampl, 0.38, label="+ clima_laboral y modalidad_trabajo", color="#9467bd")
    ax.bar_label(ax.containers[0], [f"{v:.3f}" for v in base], padding=2, fontsize=8)
    ax.bar_label(ax.containers[1], [f"{v:.3f}" for v in ampl], padding=2, fontsize=8)
    for xi, (b, a) in enumerate(zip(base, ampl)):
        d = a - b
        ax.text(xi, max(a, b) + 0.075, f"Δ = {d:+.3f}", ha="center", fontsize=8.6,
                fontweight="bold", color="#2ca02c" if d > 0 else "#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(modelos, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("F1-Score")
    ax.set_ylim(0, 1.16)
    ax.set_title("Pregunta 6: ¿mejora el modelo al agregar dos variables organizacionales?")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(_fig(f"{prefijo}_14_pregunta6.png"))
    plt.close(fig)


def graficar_costos(res: ResultadoCaso, prefijo: str) -> None:
    """Costo económico total por modelo (FN y FP ponderados)."""
    cos = {k: v for k, v in res.extra["costos"].items() if k != "_referencia"}
    ref = res.extra["costos"]["_referencia"]
    nombres = list(cos)
    totales = [cos[n]["costo_total"] for n in nombres]
    fn_costo = [res.metricas[n]["fn"] * COSTO_FALSO_NEGATIVO for n in nombres]
    fp_costo = [res.metricas[n]["fp"] * COSTO_FALSO_POSITIVO for n in nombres]
    x = np.arange(len(nombres))
    fig, ax = plt.subplots(figsize=(10.6, 4.9))
    ax.bar(x, fn_costo, 0.55, label=f"Costo por Falsos Negativos ({MONEDA} {COSTO_FALSO_NEGATIVO:,}/unidad)",
           color="#d62728")
    ax.bar(x, fp_costo, 0.55, bottom=fn_costo,
           label=f"Costo por Falsos Positivos ({MONEDA} {COSTO_FALSO_POSITIVO:,}/unidad)", color="#ff9f43")
    ax.axhline(ref["costo_sin_modelo"], color="black", ls="--", lw=1.4)
    ax.text(len(nombres) - 0.4, ref["costo_sin_modelo"] * 1.02,
            f"Costo sin modelo = {MONEDA} {ref['costo_sin_modelo']:,.0f}", ha="right", fontsize=8.6)
    for xi, t in zip(x, totales):
        ax.text(xi, t / 2, f"{t/1e6:.1f}M", ha="center", va="center",
                fontsize=8.4, color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(nombres, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel(f"Costo total ({MONEDA})")
    ax.set_title("Análisis costo-sensitivo: ¿cuánto dinero pierde la empresa con cada modelo?")
    ax.legend(fontsize=8.6, loc="upper right")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1e6:.0f}M"))
    fig.tight_layout()
    fig.savefig(_fig(f"{prefijo}_15_costos.png"))
    plt.close(fig)


# ============================================================================
# 4. EXPORTACIÓN
# ============================================================================
def exportar_resultados(res: ResultadoCaso, prefijo: str) -> None:
    """Guarda tablas (CSV/JSON) y modelos (joblib) para el informe y GitHub."""
    tabla = res.tabla_metricas()
    tabla.to_csv(DIR_DATA / f"{prefijo}_metricas_modelos.csv", index=False)
    print("\n" + tabla.round(3).to_string(index=False))

    detalle = {}
    for nom, m in res.metricas.items():
        detalle[nom] = {k: (v if not isinstance(v, (np.floating, np.integer)) else float(v))
                        for k, v in m.items() if k != "reporte"}
        detalle[nom]["reporte"] = m["reporte"]

    payload = {
        "caso": res.nombre,
        "target": res.target,
        "n_total": int(len(res.df)),
        "n_train": int(len(res.y_train)),
        "n_test": int(len(res.y_test)),
        "features": res.features,
        "tasa_clase_positiva_total": round(float(res.df[res.target].mean()), 4),
        "tasa_clase_positiva_test": round(float(res.y_test.mean()), 4),
        "baseline_accuracy": res.extra["baseline_accuracy"],
        "metricas": detalle,
        "importancia": res.extra["importancia"],
        "grid_search": res.extra["grid_search"],
        "umbrales": res.extra["umbrales"].to_dict(orient="records"),
        "costos": res.extra["costos"],
        "mejor_modelo_f1": res.extra["mejor_modelo_f1"],
    }
    if "pregunta6" in res.extra:
        payload["pregunta6"] = res.extra["pregunta6"]
    (DIR_DATA / f"{prefijo}_resultados.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    # Reporte de clasificación completo (texto plano)
    with open(DIR_DATA / f"{prefijo}_classification_report.txt", "w", encoding="utf-8") as fh:
        for nom, m in res.metricas.items():
            fh.write(f"\n{'=' * 70}\n{nom}\n{'=' * 70}\n{m['reporte']}\n")
            fh.write(f"TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}\n")

    for nom, modelo in res.modelos.items():
        joblib.dump(modelo, DIR_DATA / f"{prefijo}_modelo_{nom.replace(' ', '_').lower()}.joblib")


def ejecutar_todo() -> dict:
    """Ejecuta los dos casos de estudio completos."""
    from generar_datos import generar_caso_produccion, generar_caso_rrhh

    out = {}

    # -------- CASO 1: PRODUCCIÓN --------
    df_prod = pd.read_csv(DIR_DATA / "caso_produccion.csv")
    res_prod = ejecutar_caso("Producción", df_prod, "defectuoso",
                             {0: "No defectuoso", 1: "Defectuoso"}, "prod")
    graficar_eda(res_prod, "prod")
    graficar_modelos(res_prod, "prod")
    graficar_arbol(res_prod, "prod")
    graficar_hiperparams(res_prod, "prod")
    graficar_costos(res_prod, "prod")
    exportar_resultados(res_prod, "prod")
    out["produccion"] = res_prod

    # -------- CASO 2: RRHH --------
    df_rrhh = pd.read_csv(DIR_DATA / "caso_rrhh.csv")
    base_features = ["edad", "salario_log10", "anos_permanencia", "satisfaccion", "horas_extras"]
    res_rrhh = ejecutar_caso(
        "Rotación de empleados", df_rrhh, "renuncia",
        {0: "Permanece", 1: "Renuncia"}, "rrhh",
        features=base_features,
        features_extra=["clima_laboral", "modalidad_trabajo"],
    )
    graficar_eda(res_rrhh, "rrhh")
    graficar_modelos(res_rrhh, "rrhh")
    graficar_arbol(res_rrhh, "rrhh")
    graficar_hiperparams(res_rrhh, "rrhh")
    graficar_pregunta6(res_rrhh, "rrhh")
    exportar_resultados(res_rrhh, "rrhh")
    out["rrhh"] = res_rrhh

    return out


if __name__ == "__main__":
    ejecutar_todo()
    print("\n\n=== ANÁLISIS COMPLETO. Figuras en:", DIR_FIGS, "===")
