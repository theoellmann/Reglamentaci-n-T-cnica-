"""Generador de datos sintéticos de operaciones de importación bajo el régimen
argentino de reglamentos técnicos (Res. SIC 237/2024, 16/2025, 313/2025, 56/2026).

Simula el universo del modelo: operaciones con destino comercialización en el
mercado interno, con la respuesta jurada sobre RT del sistema MALVINA, los
datos del certificado de respaldo de la DJC, el marcado QR, alertas
internacionales y el canal de venta posterior.

La variable latente `no_conforme` (incumplimiento técnico real del producto)
se genera a partir de causas estructurales (tipo de operador, canal, categoría,
marca, respaldo documental) y luego se generan los SÍNTOMAS observables
(QR muerto, alerta internacional, certificado vencido, precio anómalo)
condicionados al incumplimiento — como ocurre en la realidad: la evidencia es
consecuencia del producto trucho, no su causa.
"""

import numpy as np
import pandas as pd

RNG_SEED = 20260611

CATEGORIAS = {
    # categoria: (peso, alcanzado_rt, riesgo_base_categoria)
    "electrico": (0.22, True, 0.55),        # Res. 16/2025
    "juguetes": (0.12, True, 0.70),         # Res. 313/2025
    "textil_calzado": (0.20, True, 0.35),   # etiquetado 549/2021 y 465/2018
    "epp": (0.06, True, 0.45),              # Res. 18/2025
    "construccion": (0.10, True, 0.30),     # Res. 236/2024
    "no_alcanzado": (0.30, False, 0.0),
}

ORIGENES = ["china", "sudeste_asiatico", "mercosur", "ue_usa", "otros"]
P_ORIGEN = [0.46, 0.14, 0.16, 0.16, 0.08]

CANALES = ["retail_formal", "mayorista", "marketplace_local", "marketplace_vendedor_ext"]

RESPUESTAS_SIM = [
    "cert_nacional", "cert_extranjero", "ensayo_laboratorio",
    "djc_simple", "sdu_muestra", "no_alcanzada_exceptuada",
]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generar_importadores(n_importadores: int, rng: np.random.Generator) -> pd.DataFrame:
    """Tres arquetipos: establecido, nuevo legítimo y oportonista (cuit de paja /
    importador golondrina), con distinta propensión estructural al incumplimiento."""
    arquetipo = rng.choice(
        ["establecido", "nuevo_legitimo", "oportunista"],
        size=n_importadores, p=[0.60, 0.25, 0.15],
    )
    antiguedad = np.where(
        arquetipo == "establecido", rng.gamma(6, 2.0, n_importadores),
        np.where(arquetipo == "nuevo_legitimo", rng.gamma(1.5, 1.0, n_importadores),
                 rng.gamma(0.8, 0.8, n_importadores)),
    ).round(1)
    cef_ratio = np.where(  # volumen importado / capacidad económica (CEF)
        arquetipo == "oportunista", rng.lognormal(1.0, 0.6, n_importadores),
        rng.lognormal(0.0, 0.4, n_importadores),
    )
    return pd.DataFrame({
        "importador_id": np.arange(n_importadores),
        "arquetipo": arquetipo,
        "antiguedad_anios": antiguedad,
        "cef_ratio": cef_ratio.round(3),
    })


def generar_operaciones(n_ops: int = 120_000, n_importadores: int = 2_500,
                        seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    imp = generar_importadores(n_importadores, rng)

    # Los oportunistas concentran más operaciones chicas y frecuentes
    pesos = np.where(imp["arquetipo"] == "oportunista", 1.6,
                     np.where(imp["arquetipo"] == "establecido", 1.2, 1.0))
    p_imp = pesos / pesos.sum()
    idx = rng.choice(n_importadores, size=n_ops, p=p_imp)
    df = imp.loc[idx].reset_index(drop=True)
    df["operacion_id"] = np.arange(n_ops)
    df["mes"] = rng.integers(1, 25, n_ops)  # 24 meses de historia

    cats = list(CATEGORIAS)
    p_cat = np.array([CATEGORIAS[c][0] for c in cats])
    df["categoria"] = rng.choice(cats, size=n_ops, p=p_cat / p_cat.sum())
    df["alcanzado_rt"] = df["categoria"].map({c: CATEGORIAS[c][1] for c in cats})

    df["origen"] = rng.choice(ORIGENES, size=n_ops, p=P_ORIGEN)

    # Canal de venta posterior: los oportunistas venden más por marketplace
    p_canal = np.zeros((n_ops, 4))
    es_oport = (df["arquetipo"] == "oportunista").to_numpy()
    p_canal[es_oport] = [0.10, 0.15, 0.30, 0.45]
    p_canal[~es_oport] = [0.45, 0.30, 0.20, 0.05]
    u = rng.random(n_ops)
    acum = p_canal.cumsum(axis=1)
    df["canal_venta"] = [CANALES[int(np.searchsorted(acum[i], u[i]))] for i in range(n_ops)]

    df["marca_conocida"] = rng.random(n_ops) < np.where(es_oport, 0.10, 0.55)

    # --- Respuesta jurada sobre RT en el SIM (solo si la NCM está alcanzada) ---
    df["respuesta_sim"] = "no_alcanzada_exceptuada"
    alcanzado = df["alcanzado_rt"].to_numpy()
    n_alc = int(alcanzado.sum())
    p_resp_normal = [0.28, 0.30, 0.12, 0.22, 0.03, 0.05]
    p_resp_oport = [0.10, 0.30, 0.10, 0.25, 0.05, 0.20]  # abusan de "exceptuada"
    resp = np.where(
        es_oport[alcanzado],
        rng.choice(RESPUESTAS_SIM, n_alc, p=p_resp_oport),
        rng.choice(RESPUESTAS_SIM, n_alc, p=p_resp_normal),
    )
    df.loc[alcanzado, "respuesta_sim"] = resp

    # Elusión flagrante: producto de uso doméstico declarado "exceptuado por uso
    # idóneo" (inválido según Anexo V Res. 16/2025). Solo la cometen oportunistas.
    domestico = df["categoria"].isin(["electrico", "juguetes"]).to_numpy()
    declara_exc = (df["respuesta_sim"] == "no_alcanzada_exceptuada").to_numpy()
    df["uso_idoneo_domestico"] = domestico & declara_exc & es_oport & (rng.random(n_ops) < 0.7)

    # --- Respaldo documental del certificado/DJC ---
    con_cert = df["respuesta_sim"].isin(["cert_nacional", "cert_extranjero"]).to_numpy()
    df["emisor_score"] = np.nan  # calidad del organismo emisor (0-1)
    df.loc[con_cert, "emisor_score"] = np.clip(
        rng.beta(5, 2, int(con_cert.sum())) - es_oport[con_cert] * 0.25, 0.05, 1.0
    ).round(3)
    df["cert_reutilizado_n"] = 0  # productos distintos amparados por el mismo cert
    df.loc[con_cert, "cert_reutilizado_n"] = np.where(
        es_oport[con_cert],
        rng.poisson(4.0, int(con_cert.sum())),
        rng.poisson(0.6, int(con_cert.sum())),
    )

    # --- Precio unitario relativo a la mediana de la NCM (se ajusta luego) ---
    df["precio_ratio"] = rng.lognormal(0.0, 0.30, n_ops).round(3)

    # ---------------------------------------------------------------
    # VARIABLE LATENTE: incumplimiento técnico real del producto
    # ---------------------------------------------------------------
    riesgo_cat = df["categoria"].map({c: CATEGORIAS[c][2] for c in cats}).to_numpy()
    logit = (
        -4.1
        + 2.0 * es_oport
        + 1.1 * riesgo_cat
        + 1.5 * (df["canal_venta"] == "marketplace_vendedor_ext").to_numpy()
        + 0.6 * (df["canal_venta"] == "marketplace_local").to_numpy()
        - 1.1 * df["marca_conocida"].to_numpy()
        + 0.35 * (df["origen"] == "china").to_numpy()          # modesto y confundido con canal
        + 0.25 * (df["origen"] == "sudeste_asiatico").to_numpy()
        + 0.9 * (df["respuesta_sim"] == "djc_simple").to_numpy() * es_oport
        + 0.8 * np.clip(df["cert_reutilizado_n"].to_numpy() - 2, 0, 6) / 3.0
        - 1.0 * np.nan_to_num(df["emisor_score"].to_numpy() - 0.6, nan=0.0)
        + 0.5 * np.clip(df["cef_ratio"].to_numpy() - 1.5, 0, 3)
        - 0.04 * df["antiguedad_anios"].to_numpy()
    )
    p_nc = _sigmoid(logit)
    df["no_conforme"] = (rng.random(n_ops) < p_nc) & alcanzado  # solo tiene sentido si hay RT

    nc = df["no_conforme"].to_numpy()

    # ---------------------------------------------------------------
    # SÍNTOMAS observables, condicionados al incumplimiento real
    # ---------------------------------------------------------------
    # QR del marcado de conformidad muerto o inexistente
    df["qr_valido"] = True
    df.loc[alcanzado, "qr_valido"] = np.where(
        nc[alcanzado], rng.random(n_alc) > 0.28, rng.random(n_alc) > 0.03
    )
    # Producto/marca alertado en Safety Gate / CPSC / GlobalRecalls
    df["alerta_internacional"] = np.where(
        nc, rng.random(n_ops) < 0.09, rng.random(n_ops) < 0.006
    )
    # Certificado vencido al momento de la verificación
    df["cert_vencido"] = False
    df.loc[con_cert, "cert_vencido"] = np.where(
        nc[con_cert], rng.random(int(con_cert.sum())) < 0.22,
        rng.random(int(con_cert.sum())) < 0.05,
    )
    # El producto trucho también se declara barato (correlación a validar: acá
    # la simulamos moderada y ruidosa, no determinista)
    df.loc[nc, "precio_ratio"] = (
        df.loc[nc, "precio_ratio"] * rng.lognormal(-0.22, 0.30, int(nc.sum()))
    ).round(3)
    # La elusión flagrante implica incumplimiento casi seguro
    flag = df["uso_idoneo_domestico"].to_numpy()
    df.loc[flag, "no_conforme"] = rng.random(int(flag.sum())) < 0.93

    # Sanciones previas en RENAI: dependen del historial real de cada importador
    tasa_nc = df.groupby("importador_id")["no_conforme"].transform("mean")
    df["sanciones_renai"] = rng.poisson(np.clip(tasa_nc * 2.2, 0, 1.5))

    cols = [
        "operacion_id", "mes", "importador_id", "arquetipo", "antiguedad_anios",
        "cef_ratio", "categoria", "alcanzado_rt", "origen", "canal_venta",
        "marca_conocida", "respuesta_sim", "uso_idoneo_domestico", "emisor_score",
        "cert_reutilizado_n", "cert_vencido", "precio_ratio", "qr_valido",
        "alerta_internacional", "sanciones_renai", "no_conforme",
    ]
    return df[cols]


if __name__ == "__main__":
    df = generar_operaciones()
    print(df.shape)
    print(df[df.alcanzado_rt]["no_conforme"].mean())
