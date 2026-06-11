"""Etapa 2: modelo supervisado (gradient boosting) entrenado sobre los
resultados de fiscalizaciones, con simulación del problema de etiquetas
selectivas (selective labels, Lakkaraju et al. 2017) y de la banda de
inspecciones aleatorias prevista en el Protocolo de Vigilancia de Mercado
(Res. SIC 56/2026).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

FEATURES_NUM = [
    "antiguedad_anios", "cef_ratio", "emisor_score", "cert_reutilizado_n",
    "precio_ratio", "sanciones_renai", "score_reglas",
]
FEATURES_BOOL = [
    "marca_conocida", "uso_idoneo_domestico", "cert_vencido", "qr_valido",
    "alerta_internacional",
]
FEATURES_CAT = ["categoria", "origen", "canal_venta", "respuesta_sim"]


def preparar_matriz(df: pd.DataFrame):
    X = pd.DataFrame(index=df.index)
    for c in FEATURES_NUM:
        X[c] = df[c].astype(float)
    for c in FEATURES_BOOL:
        X[c] = df[c].astype(float)
    for c in FEATURES_CAT:
        X[c] = df[c].astype("category")
    return X


def entrenar(df_train: pd.DataFrame, etiqueta: str = "no_conforme", seed: int = 7):
    X = preparar_matriz(df_train)
    y = df_train[etiqueta].astype(int)
    modelo = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_depth=6,
        categorical_features=[X.columns.get_loc(c) for c in FEATURES_CAT],
        class_weight="balanced", random_state=seed,
    )
    modelo.fit(X, y)
    return modelo


def precision_recall_at_k(scores: np.ndarray, y_true: np.ndarray, frac: float):
    """precision@k y recall@k con k = frac de la población (capacidad de
    inspección limitada)."""
    k = max(1, int(len(scores) * frac))
    orden = np.argsort(-scores)[:k]
    tp = y_true[orden].sum()
    return tp / k, tp / max(1, y_true.sum())


def simular_etiquetado(df_hist: pd.DataFrame, prioridad_heuristica: pd.Series,
                       cap_heuristica: float = 0.05, cap_aleatoria: float = 0.01,
                       seed: int = 11):
    """Simula qué operaciones históricas fueron inspeccionadas (y por ende
    tienen etiqueta): un 5% elegido por la heurística vieja + una banda
    aleatoria del 1% (Protocolo Res. 56/2026)."""
    rng = np.random.default_rng(seed)
    n = len(df_hist)
    k_h = int(n * cap_heuristica)
    elegidos_h = np.zeros(n, dtype=bool)
    elegidos_h[np.argsort(-prioridad_heuristica.to_numpy())[:k_h]] = True
    elegidos_a = rng.random(n) < cap_aleatoria
    return elegidos_h, elegidos_a


def importancias(modelo, df_eval: pd.DataFrame, etiqueta: str = "no_conforme",
                 seed: int = 5, n_repeats: int = 5) -> pd.Series:
    X = preparar_matriz(df_eval)
    y = df_eval[etiqueta].astype(int)
    r = permutation_importance(
        modelo, X, y, n_repeats=n_repeats, random_state=seed,
        scoring="average_precision", n_jobs=-1,
    )
    return pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)
