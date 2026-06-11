"""Pipeline completo: genera datos sintéticos, aplica la Etapa 1 (reglas),
entrena la Etapa 2 (gradient boosting) bajo etiquetas selectivas, evalúa
estrategias de selección de inspecciones y produce el informe PDF.

Uso:  python3 run_pipeline.py
Salidas en ./outputs/: metricas.json, datos de figuras e informe_modelo_rt.pdf
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score

from src.generar_datos import generar_operaciones
from src.etapa1_reglas import aplicar_reglas, heuristica_historica
from src.etapa2_modelo import (
    entrenar, preparar_matriz, precision_recall_at_k, simular_etiquetado,
    importancias,
)
from src.informe import generar_informe

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

CAPACIDADES = [0.01, 0.05]  # fracción de operaciones que se pueden fiscalizar


def main():
    print(">> Generando datos sintéticos…")
    df = generar_operaciones(n_ops=120_000, n_importadores=2_500)

    # Universo del modelo: operaciones con destino comercialización alcanzadas
    # por un RT vigente (las exceptuadas/uso personal quedan fuera, salvo la
    # inconsistencia flagrante de uso idóneo que es en sí una infracción).
    universo = df[df["alcanzado_rt"]].copy()
    base_rate = universo["no_conforme"].mean()
    print(f"   Operaciones: {len(df):,} | Universo alcanzado por RT: {len(universo):,}")
    print(f"   Tasa real de no conformidad (oculta): {base_rate:.1%}")

    print(">> Etapa 1: aplicando reglas deterministas…")
    universo = aplicar_reglas(universo)

    # Partición temporal: 18 meses de historia (entrenamiento) / 6 de evaluación
    hist = universo[universo["mes"] <= 18].reset_index(drop=True)
    test = universo[universo["mes"] > 18].reset_index(drop=True)

    print(">> Simulando etiquetas selectivas (heurística 5% + banda aleatoria 1%)…")
    prio_h = heuristica_historica(hist)
    sel_h, sel_a = simular_etiquetado(hist, prio_h)
    train_sesgado = hist[sel_h]
    train_completo = hist[sel_h | sel_a]
    print(f"   Etiquetas heurística: {len(train_sesgado):,} "
          f"(tasa observada {train_sesgado['no_conforme'].mean():.1%})")
    print(f"   + banda aleatoria: {len(train_completo):,} "
          f"(tasa banda aleatoria {hist[sel_a]['no_conforme'].mean():.1%})")

    print(">> Etapa 2: entrenando gradient boosting…")
    modelo_sesgado = entrenar(train_sesgado)
    modelo_completo = entrenar(train_completo)

    # ------------------------------------------------------------------
    # Evaluación en los 6 meses finales con la verdad oculta
    # ------------------------------------------------------------------
    print(">> Evaluando estrategias de selección en el período de prueba…")
    y_test = test["no_conforme"].to_numpy().astype(int)
    rng = np.random.default_rng(99)

    scores = {
        "Azar": rng.random(len(test)),
        "Heurística actual\n(China+categoría+nuevo)": heuristica_historica(test).to_numpy(),
        "Etapa 1: reglas": test["score_reglas"].to_numpy()
        + rng.random(len(test)) * 0.01,
        "Etapa 2: GBM\n(etiquetas sesgadas)": modelo_sesgado.predict_proba(
            preparar_matriz(test))[:, 1],
        "Etapa 2: GBM\n(sesgadas + banda aleatoria)": modelo_completo.predict_proba(
            preparar_matriz(test))[:, 1],
    }

    metricas = {}
    for nombre, s in scores.items():
        metricas[nombre] = {}
        for frac in CAPACIDADES:
            p, r = precision_recall_at_k(s, y_test, frac)
            metricas[nombre][f"precision@{frac:.0%}"] = round(float(p), 4)
            metricas[nombre][f"recall@{frac:.0%}"] = round(float(r), 4)
        metricas[nombre]["average_precision"] = round(
            float(average_precision_score(y_test, s)), 4)
        print(f"   {nombre.replace(chr(10), ' ')}: "
              + ", ".join(f"{k}={v}" for k, v in metricas[nombre].items()))

    # Curva precision-recall del mejor modelo
    s_best = scores["Etapa 2: GBM\n(sesgadas + banda aleatoria)"]
    prec, rec, _ = precision_recall_curve(y_test, s_best)

    # Importancia de variables (permutación, sobre el período de prueba)
    print(">> Calculando importancia de variables…")
    imp = importancias(modelo_completo, test)

    # Lift por decil del score
    deciles = pd.qcut(pd.Series(s_best).rank(method="first"), 10, labels=False)
    lift = (test.groupby(deciles)["no_conforme"].mean() / base_rate).tolist()

    # Tasa de no conformidad por segmento (para el informe)
    seg_canal = test.groupby("canal_venta")["no_conforme"].mean().sort_values(ascending=False)
    seg_cat = test.groupby("categoria")["no_conforme"].mean().sort_values(ascending=False)
    seg_resp = test.groupby("respuesta_sim")["no_conforme"].mean().sort_values(ascending=False)

    # Rendimiento individual de cada regla de Etapa 1 (precisión de la regla)
    reglas_cols = [c for c in test.columns if c.startswith("regla_")]
    rendimiento_reglas = {
        c.replace("regla_", ""): {
            "cobertura": round(float(test[c].mean()), 4),
            "precision": round(float(test.loc[test[c], "no_conforme"].mean()), 4)
            if test[c].any() else None,
        }
        for c in reglas_cols
    }

    resultados = {
        "n_operaciones": int(len(df)),
        "n_universo": int(len(universo)),
        "n_test": int(len(test)),
        "tasa_base": round(float(base_rate), 4),
        "tasa_base_test": round(float(y_test.mean()), 4),
        "n_etiquetas_sesgadas": int(len(train_sesgado)),
        "n_etiquetas_completas": int(len(train_completo)),
        "tasa_obs_heuristica": round(float(train_sesgado["no_conforme"].mean()), 4),
        "tasa_banda_aleatoria": round(float(hist[sel_a]["no_conforme"].mean()), 4),
        "metricas": metricas,
        "pr_curve": {"precision": prec[::20].tolist(), "recall": rec[::20].tolist()},
        "importancias": {k: round(float(v), 5) for k, v in imp.items()},
        "lift_deciles": [round(float(x), 2) for x in lift],
        "seg_canal": {k: round(float(v), 4) for k, v in seg_canal.items()},
        "seg_categoria": {k: round(float(v), 4) for k, v in seg_cat.items()},
        "seg_respuesta_sim": {k: round(float(v), 4) for k, v in seg_resp.items()},
        "rendimiento_reglas": rendimiento_reglas,
    }
    (OUT / "metricas.json").write_text(json.dumps(resultados, indent=2, ensure_ascii=False))
    print(f">> Métricas guardadas en {OUT/'metricas.json'}")

    print(">> Generando informe PDF…")
    pdf_path = OUT / "informe_modelo_rt.pdf"
    generar_informe(resultados, pdf_path)
    print(f">> Informe: {pdf_path}")


if __name__ == "__main__":
    main()
