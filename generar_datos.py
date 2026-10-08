"""
generar_datos.py
================
Construcción de los dos datasets sintéticos del caso de estudio, de acuerdo
con las relaciones causales descritas en la guía
"UC_MLS_IMPL_ALG_STUDY_CASE.pdf" (Universidad de Cundinamarca, 2026).

Caso 1 - Producción (XYZ Inc.)
    Variables: temperatura, presion, velocidad, horas_operacion, vibracion
    Objetivo : defectuoso (1) / no defectuoso (0)

Caso 2 - RR. HH. (Rotación de empleados)
    Variables: edad, salario, anos_permanencia, satisfaccion, horas_extras
    Objetivo : renuncia (1) / permanece (0)
    Extra    : clima_laboral, modalidad_trabajo  (pregunta 6 de la guía)

Los datos se generan con un Mecanismo Generador de Datos (DGP) basado en un
modelo logístico latente + ruido, de modo que:
  * las variables tengan correlaciones realistas entre sí;
  * el problema sea aprendible pero NO trivial (accuracy techo ~0.90);
  * exista desbalance de clases moderado (caso realista de industria);
  * el proceso sea 100 % reproducible (semilla fija).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (DIR_DATA, N_CASO_PRODUCCION, N_CASO_RRHH, SEED)  # noqa: E402


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Función sigmoide numéricamente estable."""
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def _calibrar_intercept(z: np.ndarray, tasa_objetivo: float, rng) -> np.ndarray:
    """Ajusta el intercept por bisección para lograr la tasa de clase 1 deseada.

    Permite controlar el nivel de desbalance del dataset sin alterar las
    relaciones (pendientes) entre las variables predictoras y el objetivo.
    """
    lo, hi = -30.0, 30.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        muestra = rng.binomial(1, _sigmoid(z + mid))
        tasa = muestra.mean()
        if tasa < tasa_objetivo:
            lo = mid
        else:
            hi = mid
    return z + (lo + hi) / 2.0


# ----------------------------------------------------------------------------
# CASO 1: PREDICCIÓN DE DEFECTOS EN PRODUCCIÓN
# ----------------------------------------------------------------------------
def generar_caso_produccion(n: int = N_CASO_PRODUCCION, seed: int = SEED) -> pd.DataFrame:
    """Genera el dataset de lotes de producción de la empresa XYZ Inc.

    Lógica del proceso (DGP):
      * `temperatura` fuera de la ventana óptima 78-85 °C aumenta el defecto
        (término cuadrático centrado en 81.5).
      * `presion` y `velocidad` altas fuerzan la máquina -> más defectos.
      * `horas_operacion` acumuladas generan desgaste -> más defectos.
      * `vibracion` es el síntoma más directo de desalineación -> coeficiente
        dominante.
      * Interacción velocidad x vibración: a alta velocidad la vibración
        castiga mucho más la calidad.
    """
    rng = np.random.default_rng(seed)

    horas_operacion = np.round(rng.uniform(0.5, 24.0, n), 1)
    # La temperatura sube con las horas de operación continua (deriva térmica)
    temperatura = np.round(
        rng.normal(82.0, 3.2, n) + 0.16 * horas_operacion + rng.normal(0, 0.6, n), 1
    )
    presion = np.round(rng.normal(120.0, 9.0, n) + 0.25 * horas_operacion, 1)
    velocidad = np.round(rng.normal(100.0, 12.0, n) - 0.12 * horas_operacion, 1)
    # La vibración crece con velocidad, presión y desgaste acumulado
    vibracion = np.round(
        1.2
        + 0.022 * (velocidad - 100)
        + 0.018 * (presion - 120)
        + 0.035 * horas_operacion
        + rng.normal(0, 0.45, n),
        2,
    )
    vibracion = np.clip(vibracion, 0.1, None)

    z = (
        0.062 * (temperatura - 82.0) ** 2      # desviación térmica (efecto en U)
        + 0.052 * (presion - 120.0)            # sobrepresión del sistema
        + 0.060 * (velocidad - 100.0)          # exceso de velocidad de línea
        + 0.105 * horas_operacion              # desgaste acumulado
        + 1.60 * vibracion                     # síntoma dominante de desalineación
        + 0.0016 * (velocidad - 100) * (vibracion - 2.0)   # interacción
    )
    # Se calibra el intercept para una tasa de defectos realista (~17 %)
    z = _calibrar_intercept(z, 0.17, np.random.default_rng(seed + 99))
    p_defecto = _sigmoid(z)
    defectuoso = rng.binomial(1, p_defecto)

    lote = np.arange(1, n + 1)
    df = pd.DataFrame(
        {
            "lote": lote,
            "temperatura": temperatura,
            "presion": presion,
            "velocidad": velocidad,
            "horas_operacion": horas_operacion,
            "vibracion": vibracion,
            "defectuoso": defectuoso.astype(int),
        }
    )
    return df


# ----------------------------------------------------------------------------
# CASO 2: ROTACIÓN DE EMPLEADOS
# ----------------------------------------------------------------------------
def generar_caso_rrhh(n: int = N_CASO_RRHH, seed: int = SEED) -> pd.DataFrame:
    """Genera el dataset de rotación de personal.

    Lógica del proceso (DGP), coherente con las sospechas del enunciado:
      * baja satisfacción  -> mayor renuncia (coeficiente dominante);
      * muchas horas extras -> mayor renuncia;
      * pocos años de permanencia -> mayor renuncia;
      * salario bajo -> mayor renuncia (escala logarítmica: importa el
        salario relativo, no el absoluto);
      * edad -> efecto leve en U (los muy jóvenes rotan más).
    Variables añadidas para la pregunta 6:
      * `clima_laboral` (escala 1-10) y `modalidad_trabajo`
        (0 = presencial, 1 = híbrida, 2 = remota).
    """
    rng = np.random.default_rng(seed + 1)

    edad = np.clip(rng.normal(36.0, 8.5, n), 21, 63).round(0).astype(int)
    anos_permanencia = np.clip(
        rng.gamma(shape=1.55, scale=3.4, size=n) - 0.5, 0.3, 32.0
    ).round(1)
    # El salario crece con antigüedad y edad (realista y correlacionado)
    salario = np.clip(
        rng.normal(2_900_000, 750_000, n)
        + 92_000 * anos_permanencia
        + 17_000 * (edad - 36),
        1_300_000,
        9_500_000,
    ).round(-3)
    satisfaccion = np.clip(
        rng.normal(6.4, 1.9, n)
        - 0.035 * anos_permanencia
        + 0.00000006 * salario
        - 0.16 * rng.normal(8, 2.5, n) ** 0.5,
        1.0,
        10.0,
    ).round(1)
    horas_extras = np.clip(
        rng.gamma(shape=2.2, scale=2.6, size=n) + rng.normal(0, 1.0, n), 0, 22
    ).round(1)

    # --- Variables adicionales (pregunta 6) ---------------------------------
    clima_laboral = np.clip(
        rng.normal(6.6, 1.7, n) + 0.32 * (satisfaccion - 6.4) + rng.normal(0, 0.7, n),
        1.0,
        10.0,
    ).round(1)
    modalidad_trabajo = rng.choice([0, 1, 2], size=n, p=[0.52, 0.31, 0.17])
    # El trabajo remoto/híbrido reduce levemente la renuncia (flexibilidad)
    efecto_modalidad = np.where(modalidad_trabajo == 2, -0.45,
                       np.where(modalidad_trabajo == 1, -0.22, 0.0))

    z = (
        - 1.90 * np.log(salario / 3_300_000)             # salario relativo bajo
        - 0.24 * anos_permanencia                        # poca antigüedad
        - 0.72 * (satisfaccion - 6.0)                    # insatisfacción
        + 0.21 * horas_extras                            # sobrecarga laboral
        - 0.035 * (edad - 36.0)                          # edad (efecto leve)
        - 0.42 * (clima_laboral - 6.6)                   # clima laboral
        + efecto_modalidad
    )
    # Se calibra el intercept para una tasa de rotación realista (~23 %)
    z = _calibrar_intercept(z, 0.23, np.random.default_rng(seed + 77))
    p_renuncia = _sigmoid(z)
    renuncia = rng.binomial(1, p_renuncia)

    df = pd.DataFrame(
        {
            "empleado_id": np.arange(1, n + 1),
            "edad": edad,
            "salario": salario.astype(int),
            # Transformación log10 del salario: corrige la asimetría positiva de
            # la variable y estabiliza la varianza. Es un requisito práctico del
            # clasificador Gaussiano (Naive Bayes), cuya estimación de verosimilitud
            # asume normalidad en cada predictor.
            "salario_log10": np.log10(salario).round(4),
            "anos_permanencia": anos_permanencia,
            "satisfaccion": satisfaccion,
            "horas_extras": horas_extras,
            "clima_laboral": clima_laboral,
            "modalidad_trabajo": modalidad_trabajo,
            "renuncia": renuncia.astype(int),
        }
    )
    return df


# ----------------------------------------------------------------------------
def main() -> None:
    prod = generar_caso_produccion()
    rrhh = generar_caso_rrhh()

    p_prod = DIR_DATA / "caso_produccion.csv"
    p_rrhh = DIR_DATA / "caso_rrhh.csv"
    prod.to_csv(p_prod, index=False)
    rrhh.to_csv(p_rrhh, index=False)

    for nombre, df, target in (
        ("PRODUCCIÓN", prod, "defectuoso"),
        ("RR. HH.", rrhh, "renuncia"),
    ):
        print(f"\n{'=' * 68}\nDataset {nombre}  ->  {p_prod if nombre=='PRODUCCIÓN' else p_rrhh}")
        print(f"Filas: {len(df)}  |  Columnas: {df.shape[1]}  |  Semilla: {SEED}")
        vc = df[target].value_counts(normalize=True).sort_index()
        print(f"Clase 0: {vc.get(0, 0):.3%}   Clase 1: {vc.get(1, 0):.3%}")
        print(df.describe().T[["mean", "std", "min", "max"]].round(2).to_string())


if __name__ == "__main__":
    main()
