"""Etapa 1: reglas deterministas y score de anomalías.

No requiere etiquetas de entrenamiento — son verificaciones directas contra el
régimen vigente (Res. 237/2024, 16/2025, 56/2026) y bases públicas. Es el motor
de arranque mientras la fiscalización ex post genera etiquetas.
"""

import pandas as pd

REGLAS = {
    # nombre: (descripcion, puntos)
    "uso_idoneo_domestico": ("Producto doméstico declarado 'exceptuado por uso idóneo' (inválido, Anexo V Res. 16/2025)", 5),
    "qr_muerto": ("Marcado de conformidad sin QR válido / QR no enlaza a DJC (Res. 237/2024)", 4),
    "alerta_internacional": ("Producto o marca alertado en Safety Gate / CPSC / OECD GlobalRecalls", 5),
    "cert_vencido": ("Certificado de respaldo de la DJC vencido", 3),
    "cert_sobre_reutilizado": ("Mismo certificado ampara >3 productos distintos", 2),
    "emisor_dudoso": ("Organismo emisor del certificado con score de calidad < 0.4", 2),
    "precio_anomalo": ("Precio unitario < 50% de la mediana de la NCM", 2),
    "reincidente_renai": ("Importador con sanciones previas en RENAI", 3),
    "canal_marketplace_ext": ("Venta vía marketplace con vendedor extranjero directo", 1),
}


def aplicar_reglas(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve el DataFrame con una columna booleana por regla y el score total."""
    r = pd.DataFrame(index=df.index)
    r["uso_idoneo_domestico"] = df["uso_idoneo_domestico"]
    r["qr_muerto"] = ~df["qr_valido"] & df["alcanzado_rt"]
    r["alerta_internacional"] = df["alerta_internacional"]
    r["cert_vencido"] = df["cert_vencido"]
    r["cert_sobre_reutilizado"] = df["cert_reutilizado_n"] > 3
    r["emisor_dudoso"] = df["emisor_score"].fillna(1.0) < 0.4
    r["precio_anomalo"] = df["precio_ratio"] < 0.5
    r["reincidente_renai"] = df["sanciones_renai"] > 0
    r["canal_marketplace_ext"] = df["canal_venta"] == "marketplace_vendedor_ext"

    puntos = pd.Series({k: v[1] for k, v in REGLAS.items()})
    score = (r.astype(int) * puntos).sum(axis=1)

    out = df.copy()
    for c in r.columns:
        out[f"regla_{c}"] = r[c]
    out["score_reglas"] = score
    return out


def heuristica_historica(df: pd.DataFrame) -> pd.Series:
    """Política de selección 'vieja': priorizar China + categorías sensibles +
    importador nuevo. Es la que genera las etiquetas históricas sesgadas."""
    s = (
        2 * (df["origen"] == "china").astype(int)
        + 2 * df["categoria"].isin(["juguetes", "electrico"]).astype(int)
        + 1 * (df["antiguedad_anios"] < 2).astype(int)
    )
    return s + 0.001 * pd.Series(range(len(df)), index=df.index) % 1  # desempate estable
