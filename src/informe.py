"""Genera el informe PDF del modelo de detección ex post de incumplimiento de
reglamentos técnicos, a partir de los resultados del pipeline sobre datos
sintéticos."""

import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

AZUL = "#1f4e79"
GRIS = "#444444"
NARANJA = "#d97706"


def _pagina_texto(pdf, titulo, bloques, pie=None):
    """Página A4 de texto: bloques = lista de (subtitulo|None, parrafo)."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.07, 0.95, titulo, fontsize=16, fontweight="bold", color=AZUL)
    y = 0.90
    for sub, parr in bloques:
        if sub:
            fig.text(0.07, y, sub, fontsize=11.5, fontweight="bold", color=GRIS)
            y -= 0.022
        for linea in textwrap.wrap(parr, width=102):
            fig.text(0.07, y, linea, fontsize=9.3, color="black")
            y -= 0.0165
        y -= 0.014
    if pie:
        fig.text(0.07, 0.03, pie, fontsize=7.5, color="gray")
    pdf.savefig(fig)
    plt.close(fig)


def generar_informe(r: dict, ruta_pdf):
    m = r["metricas"]
    nombres = list(m.keys())
    mejor = nombres[-1]
    p1_mejor = m[mejor]["precision@1%"]
    p5_mejor = m[mejor]["precision@5%"]
    r5_mejor = m[mejor]["recall@5%"]
    p5_azar = m["Azar"]["precision@5%"]
    p5_heur = m[nombres[1]]["precision@5%"]
    p5_reglas = m["Etapa 1: reglas"]["precision@5%"]
    p5_sesgado = m[nombres[3]]["precision@5%"]
    ap_sesgado = m[nombres[3]]["average_precision"]
    ap_completo = m[mejor]["average_precision"]

    with PdfPages(ruta_pdf) as pdf:
        # ------------------------------------------------ Portada / resumen
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.90, "Modelo de detección ex post de incumplimiento de",
                 fontsize=17, fontweight="bold", color=AZUL)
        fig.text(0.07, 0.875, "reglamentos técnicos en importaciones — Argentina",
                 fontsize=17, fontweight="bold", color=AZUL)
        fig.text(0.07, 0.845, "Prueba de concepto sobre datos sintéticos · junio 2026",
                 fontsize=11, color=GRIS)
        fig.text(0.07, 0.79, "Resumen ejecutivo", fontsize=13, fontweight="bold", color=GRIS)
        resumen = (
            f"Se construyó y evaluó una prueba de concepto del modelo de riesgo para priorizar la "
            f"fiscalización ex post de productos importados alcanzados por reglamentos técnicos, según el "
            f"régimen vigente (Res. SIC 237/2024, 16/2025, 313/2025 y Protocolo de Vigilancia de Mercado "
            f"Res. 56/2026). Se simularon {r['n_operaciones']:,} operaciones de importación de 2.500 "
            f"importadores durante 24 meses; {r['n_universo']:,} integran el universo fiscalizable "
            f"(NCM alcanzada por un RT, destino comercialización). La tasa real de no conformidad "
            f"—oculta para el modelo— es {r['tasa_base']:.1%}."
        )
        y = 0.765
        for linea in textwrap.wrap(resumen, width=100):
            fig.text(0.07, y, linea, fontsize=9.5); y -= 0.017
        y -= 0.015
        hallazgos = [
            f"Con capacidad de fiscalizar solo el 5% de las operaciones, el modelo completo (gradient "
            f"boosting + banda aleatoria) logra una precisión de {p5_mejor:.0%}: de cada 100 productos "
            f"enviados a ensayo, ~{round(p5_mejor*100)} resultan no conformes, contra {p5_azar:.0%} "
            f"si se eligiera al azar y {p5_heur:.0%} con la heurística tradicional (origen+categoría).",
            f"Al 1% de capacidad la precisión sube a {p1_mejor:.0%}, en línea con lo reportado por la "
            f"literatura (DATE/OMA, KDD 2020: 92,7% al 1% en Nigeria).",
            f"Con el 5% de capacidad se captura el {r5_mejor:.0%} de todos los productos no conformes "
            f"del período (recall@5%).",
            f"Las reglas deterministas de Etapa 1 (QR muerto, alerta internacional, certificado vencido, "
            f"uso idóneo inválido) alcanzan por sí solas {p5_reglas:.0%} de precisión sin necesidad de "
            f"etiquetas de entrenamiento: son el motor de arranque recomendado.",
            f"El sesgo de etiquetas selectivas distorsiona el diagnóstico: la muestra elegida por la "
            f"heurística vieja 've' una tasa de incumplimiento de {r['tasa_obs_heuristica']:.0%} cuando "
            f"la real es {r['tasa_base']:.0%}. La banda aleatoria del 1% (habilitada por la Res. 56/2026) "
            f"es la única fuente de estimación insesgada y mejora además el modelo "
            f"(AP {ap_sesgado:.2f} → {ap_completo:.2f}).",
        ]
        fig.text(0.07, y, "Hallazgos principales", fontsize=13, fontweight="bold", color=GRIS)
        y -= 0.025
        for h in hallazgos:
            primera = True
            for linea in textwrap.wrap(h, width=96):
                pref = "  •  " if primera else "      "
                fig.text(0.07, y, pref + linea, fontsize=9.5)
                y -= 0.017
                primera = False
            y -= 0.008
        fig.text(0.07, 0.06, "Advertencia: resultados sobre datos SINTÉTICOS construidos según la estructura del régimen argentino y la evidencia",
                 fontsize=8, color="gray")
        fig.text(0.07, 0.047, "internacional. Validan la arquitectura y la metodología, no el rendimiento sobre datos reales.",
                 fontsize=8, color="gray")
        pdf.savefig(fig); plt.close(fig)

        # ------------------------------------------------ Contexto y universo
        _pagina_texto(pdf, "1. Contexto normativo y universo del modelo", [
            ("El régimen vigente (2024–2026)",
             "La Res. SIC 237/2024 (Marco General de Evaluación de la Conformidad) trasladó el control de los "
             "reglamentos técnicos de la frontera al mercado: la conformidad se acredita con una Declaración "
             "Jurada de Conformidad (DJC) del importador/fabricante —respaldada en certificados o ensayos, "
             "locales o extranjeros (Decreto 892/2025)— y un marcado con código QR que debe enlazar a la DJC. "
             "La Aduana está dispensada del control desde febrero de 2025; en cada despacho el importador "
             "responde con carácter de declaración jurada su situación frente al RT (preguntas del sistema "
             "MALVINA). La fiscalización es ex post, en el mercado, bajo el Protocolo de Vigilancia de Mercado "
             "(Res. 56/2026), que prevé verificación documental, toma de muestras por triplicado, ensayos de "
             "laboratorio y controles aleatorios según riesgo. Sanciona la Subsecretaría de Defensa del "
             "Consumidor (Ley 24.240 y Dto. 274/2019), con registro público de infractores (RENAI)."),
            ("Universo fiscalizable",
             "El hecho alcanzado por los RT es la comercialización en el mercado interno (fórmula literal de "
             "las Res. 16/2025 y 236/2024). Quedan fuera: uso personal (courier, equipaje), repuestos e "
             "insumos, y uso profesional exclusivo (exento de certificar, no de DJC). El modelo por lo tanto "
             "no persigue importar: persigue (a) productos comercializados sin conformidad real y (b) la "
             "incoherencia entre el destino declarado y el comportamiento posterior — p. ej., declarar "
             "'exceptuado por uso idóneo' un producto de uso doméstico, lo que el Anexo V de la Res. 16/2025 "
             "prohíbe expresamente y constituye una regla determinista de detección."),
            ("Variable objetivo",
             "El target es técnico, no fiscal: producto no conforme al reglamento (riesgo de seguridad), "
             "determinado por el resultado del ensayo de laboratorio de la fiscalización. El valor declarado "
             "en aduana solo participa como señal indirecta (proxy de calidad), nunca como objetivo."),
        ])

        # ------------------------------------------------ Variables
        _pagina_texto(pdf, "2. Variables del modelo y su respaldo", [
            ("Señales documentales (verificación directa del régimen)",
             "QR del marcado de conformidad inexistente o que no enlaza a una DJC válida (Res. 237/2024 exige "
             "acceso por 10 años); certificado de respaldo vencido, de organismo de baja calidad, o reutilizado "
             "para amparar productos distintos; respuesta jurada en el SIM ('certificado vigente', 'ensayo', "
             "'DJC', 'sin derecho a uso', 'no alcanzada/exceptuada') y la sede del organismo emisor declarado."),
            ("Señales de producto y mercado",
             "Producto o marca alertado en Safety Gate (UE), CPSC (EE.UU.) u OECD GlobalRecalls — además de "
             "señal de riesgo, con el Dto. 892/2025 verifica la vía 'país de referencia'; canal de venta "
             "(marketplace con vendedor extranjero directo es el predictor mejor documentado: 66–96% de "
             "incumplimiento en campañas UE/UK contra ~2% en comercio formal certificado); categoría de "
             "producto (los diferenciados —juguetes, eléctricos, textiles— concentran el riesgo: Javorcik & "
             "Narciso 2008); marca desconocida; precio unitario anómalo respecto de la mediana de la NCM "
             "(correlación con incumplimiento técnico: hipótesis a validar, vacío explícito en la literatura)."),
            ("Señales del operador",
             "Historial de sanciones (RENAI, Disp. 362/2026) — criterio expreso del Protocolo y del art. 11 "
             "del Reg. UE 2019/1020, y el predictor operativo central del sistema análogo de la FDA; "
             "antigüedad y volumen del importador; relación volumen/capacidad económica (CEF, histórico "
             "SIRA/SEDI); identidad de alta cardinalidad del operador (Vanhoeyveld et al. 2020, 9,6M de "
             "declaraciones belgas)."),
            ("Señales de elusión",
             "Declaración SIM inconsistente con el comportamiento: 'exceptuada por uso' con venta minorista "
             "posterior; reclasificación hacia NCM vecinas fuera del alcance del RT con descripción idéntica "
             "(las medidas no arancelarias restrictivas causan por sí mismas fraude declarativo: Kee & Nicita "
             "2022, Journal of International Economics)."),
            ("Fuentes de datos reales para producción",
             "SIM/MALVINA (despachos + respuestas RT), DJC públicas vía QR/web (escrapeables), registros de "
             "certificados de los organismos de evaluación, RENAI, histórico SIRA/SEDI con CEF, bases "
             "internacionales de alertas, listados de e-commerce (precio, vendedor, reseñas — criterios "
             "documentados del OPSS británico)."),
        ])

        # ------------------------------------------------ Metodología
        _pagina_texto(pdf, "3. Metodología y simulación", [
            ("Datos sintéticos",
             f"Se generaron {r['n_operaciones']:,} operaciones (24 meses, 2.500 importadores con arquetipos "
             "establecido / nuevo / oportunista). El incumplimiento real se genera desde causas estructurales "
             "(operador, canal, categoría, marca, respaldo documental) y los síntomas observables (QR muerto, "
             "alerta internacional, certificado vencido, precio bajo) se generan condicionados al "
             "incumplimiento — la evidencia es consecuencia del producto no conforme, como en la realidad. "
             f"Tasa de no conformidad resultante en el universo alcanzado: {r['tasa_base']:.1%}."),
            ("Arquitectura en dos etapas",
             "Etapa 1 (sin etiquetas): reglas deterministas con puntaje — verificables hoy mismo contra el "
             "régimen. Etapa 2 (supervisada): gradient boosting (HistGradientBoosting) sobre todas las "
             "variables, incluida la Etapa 1 como feature, entrenado con los resultados de fiscalizaciones."),
            ("Etiquetas selectivas y banda aleatoria",
             f"Solo se conoce el resultado de lo inspeccionado (selective labels, Lakkaraju et al. 2017). Se "
             f"simuló: 18 meses de historia en los que se fiscalizó el 5% elegido por la heurística "
             f"tradicional (origen China + categoría sensible + importador nuevo) — "
             f"{r['n_etiquetas_sesgadas']:,} etiquetas con tasa observada {r['tasa_obs_heuristica']:.1%} — "
             f"más una banda aleatoria del 1% ({r['n_etiquetas_completas'] - r['n_etiquetas_sesgadas']:,} "
             f"etiquetas, tasa {r['tasa_banda_aleatoria']:.1%}, estimador insesgado de la tasa poblacional). "
             "Se entrenan dos modelos: solo-sesgado vs sesgado+banda, para medir el costo del sesgo."),
            ("Evaluación",
             f"Los 6 meses finales ({r['n_test']:,} operaciones) se reservan como período de prueba, donde la "
             "verdad oculta es conocida por el simulador. La métrica principal es precision@k (tasa de acierto "
             "dentro de la capacidad k de fiscalización, k=1% y 5%) y recall@k (porción del incumplimiento "
             "total capturada), comparando cinco estrategias: azar, heurística tradicional, reglas de Etapa 1, "
             "y el GBM con cada régimen de etiquetas."),
        ])

        # ------------------------------------------------ Resultados: barras
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        fig.suptitle("4. Resultados: precisión según estrategia de selección",
                     fontsize=15, fontweight="bold", color=AZUL, x=0.07, ha="left")
        ax.set_position([0.10, 0.55, 0.85, 0.34])
        x = range(len(nombres))
        p1 = [m[n]["precision@1%"] for n in nombres]
        p5 = [m[n]["precision@5%"] for n in nombres]
        w = 0.38
        ax.bar([i - w / 2 for i in x], p1, w, label="precision@1%", color=AZUL)
        ax.bar([i + w / 2 for i in x], p5, w, label="precision@5%", color=NARANJA)
        for i in x:
            ax.text(i - w / 2, p1[i] + 0.012, f"{p1[i]:.0%}", ha="center", fontsize=8)
            ax.text(i + w / 2, p5[i] + 0.012, f"{p5[i]:.0%}", ha="center", fontsize=8)
        ax.axhline(r["tasa_base_test"], ls="--", color="gray", lw=1)
        ax.text(len(nombres) - 0.5, r["tasa_base_test"] + 0.01,
                f"tasa base {r['tasa_base_test']:.0%}", color="gray", fontsize=8, ha="right")
        ax.set_xticks(list(x))
        ax.set_xticklabels(nombres, fontsize=8)
        ax.set_ylabel("precisión (no conformes / inspeccionados)")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

        # tabla de métricas
        tabla_y = 0.46
        fig.text(0.07, tabla_y + 0.02, "Métricas completas (período de prueba)",
                 fontsize=11, fontweight="bold", color=GRIS)
        cols = ["precision@1%", "recall@1%", "precision@5%", "recall@5%", "average_precision"]
        fig.text(0.30, tabla_y - 0.01, "  ".join(f"{c:>16}" for c in cols), fontsize=7.5,
                 family="monospace")
        yy = tabla_y - 0.030
        for n in nombres:
            fig.text(0.07, yy, n.replace("\n", " ")[:34], fontsize=7.5, family="monospace")
            fig.text(0.30, yy, "  ".join(f"{m[n][c]:>16.1%}" if c != "average_precision"
                                         else f"{m[n][c]:>16.3f}" for c in cols),
                     fontsize=7.5, family="monospace")
            yy -= 0.020
        lectura = (
            "Lectura: con la misma capacidad de inspección, pasar de la heurística tradicional al modelo "
            "completo multiplica por "
            f"{m[mejor]['precision@5%'] / max(m[nombres[1]]['precision@5%'], 1e-9):.1f} la tasa de acierto. "
            "Las reglas de Etapa 1 ya capturan la mayor parte de la ganancia sin requerir historial de "
            "fiscalizaciones, lo que valida la estrategia de arranque en dos etapas."
        )
        yy -= 0.015
        for linea in textwrap.wrap(lectura, width=102):
            fig.text(0.07, yy, linea, fontsize=9.3); yy -= 0.0165
        pdf.savefig(fig); plt.close(fig)

        # ------------------------------------------------ PR curve + lift
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle("5. Curva precisión-cobertura y concentración del riesgo",
                     fontsize=15, fontweight="bold", color=AZUL, x=0.07, ha="left")
        ax1 = fig.add_axes([0.10, 0.58, 0.85, 0.30])
        ax1.plot(r["pr_curve"]["recall"], r["pr_curve"]["precision"], color=AZUL, lw=2)
        ax1.axhline(r["tasa_base_test"], ls="--", color="gray", lw=1)
        ax1.set_xlabel("recall (porción del incumplimiento total capturada)")
        ax1.set_ylabel("precisión")
        ax1.set_title("Curva precisión-recall — modelo completo", fontsize=10, color=GRIS)
        ax1.spines[["top", "right"]].set_visible(False)

        ax2 = fig.add_axes([0.10, 0.16, 0.85, 0.30])
        lift = r["lift_deciles"]
        ax2.bar(range(1, 11), lift, color=[NARANJA if i == 10 else AZUL for i in range(1, 11)])
        ax2.axhline(1, ls="--", color="gray", lw=1)
        for i, v in enumerate(lift, 1):
            ax2.text(i, v + 0.1, f"{v:.1f}x", ha="center", fontsize=8)
        ax2.set_xlabel("decil del score (10 = mayor riesgo)")
        ax2.set_ylabel("lift vs tasa base")
        ax2.set_title("Lift por decil: cuántas veces más incumplimiento concentra cada decil",
                      fontsize=10, color=GRIS)
        ax2.spines[["top", "right"]].set_visible(False)
        fig.text(0.07, 0.07, textwrap.fill(
            f"El decil superior concentra {lift[-1]:.1f} veces la tasa base de incumplimiento; los deciles "
            "inferiores quedan prácticamente limpios, lo que habilita liberar capacidad de inspección sin "
            "costo (evidencia análoga: Albania, World Bank Economic Review 2021 — reducir inspecciones de "
            "bajo riesgo no aumentó la evasión).", width=105), fontsize=9)
        pdf.savefig(fig); plt.close(fig)

        # ------------------------------------------------ Importancias + reglas
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle("6. Qué variables explican la detección",
                     fontsize=15, fontweight="bold", color=AZUL, x=0.07, ha="left")
        imp = dict(list(r["importancias"].items())[:14])
        ax1 = fig.add_axes([0.32, 0.55, 0.62, 0.36])
        ax1.barh(list(imp.keys())[::-1], list(imp.values())[::-1], color=AZUL)
        ax1.set_xlabel("importancia por permutación (Δ average precision)")
        ax1.tick_params(labelsize=8)
        ax1.spines[["top", "right"]].set_visible(False)

        fig.text(0.07, 0.48, "Rendimiento individual de las reglas de Etapa 1 (período de prueba)",
                 fontsize=11, fontweight="bold", color=GRIS)
        fig.text(0.07, 0.455, f"{'regla':<28}{'cobertura':>12}{'precisión':>12}",
                 fontsize=8.5, family="monospace", fontweight="bold")
        yy = 0.435
        rr = sorted(r["rendimiento_reglas"].items(),
                    key=lambda kv: -(kv[1]["precision"] or 0))
        for nombre, v in rr:
            prec_txt = f"{v['precision']:.0%}" if v["precision"] is not None else "—"
            fig.text(0.07, yy, f"{nombre:<28}{v['cobertura']:>11.1%}{prec_txt:>12}",
                     fontsize=8.5, family="monospace")
            yy -= 0.019
        yy -= 0.02
        for linea in textwrap.wrap(
            "Las señales documentales directas (alerta internacional, QR muerto, uso idóneo inválido, "
            "certificado vencido) son a la vez las más importantes para el modelo y las de mayor precisión "
            "individual: confirman que la verificación automática del respaldo documental es el corazón del "
            "sistema. Las variables del operador (sanciones RENAI, antigüedad, CEF) aportan poder adicional "
            "para los casos sin síntoma visible. El origen aporta poco una vez controlado el canal y la "
            "marca — consistente con la advertencia de Bapuji & Beamish (2008) sobre usar país de origen "
            "como predictor crudo.", width=105):
            fig.text(0.07, yy, linea, fontsize=9.3); yy -= 0.0165
        pdf.savefig(fig); plt.close(fig)

        # ------------------------------------------------ Segmentos
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle("7. Tasa de no conformidad por segmento (verdad del simulador)",
                     fontsize=15, fontweight="bold", color=AZUL, x=0.07, ha="left")
        for i, (titulo, datos) in enumerate([
            ("por canal de venta", r["seg_canal"]),
            ("por categoría de producto", r["seg_categoria"]),
            ("por respuesta jurada en el SIM", r["seg_respuesta_sim"]),
        ]):
            ax = fig.add_axes([0.34, 0.66 - i * 0.27, 0.60, 0.18])
            ks = list(datos.keys())[::-1]
            vs = [datos[k] for k in ks]
            ax.barh(ks, vs, color=NARANJA)
            for j, v in enumerate(vs):
                ax.text(v + 0.005, j, f"{v:.0%}", va="center", fontsize=8)
            ax.set_title(titulo, fontsize=10, color=GRIS, loc="left")
            ax.tick_params(labelsize=8)
            ax.set_xlim(0, max(vs) * 1.25)
            ax.spines[["top", "right"]].set_visible(False)
        fig.text(0.07, 0.06, textwrap.fill(
            "El gradiente por canal reproduce la evidencia internacional (marketplace con vendedor "
            "extranjero >> comercio formal). La respuesta 'no alcanzada/exceptuada' con producto en góndola "
            "y la DJC simple sin certificado concentran el riesgo declarativo.", width=105), fontsize=9)
        pdf.savefig(fig); plt.close(fig)

        # ------------------------------------------------ Selective labels + cierre
        _pagina_texto(pdf, "8. Etiquetas selectivas, limitaciones y próximos pasos", [
            ("Costo medido del sesgo de selección",
             f"En una sola ronda de entrenamiento la brecha es modesta: el modelo entrenado únicamente con "
             f"inspecciones de la heurística tradicional alcanza AP = {ap_sesgado:.2f} (precision@5% = "
             f"{p5_sesgado:.0%}), contra AP = {ap_completo:.2f} (precision@5% = {p5_mejor:.0%}) al sumar la "
             "banda aleatoria del 1%. "
             "La brecha es chica porque las señales documentales (QR, alertas, certificados) son globales y "
             "se aprenden incluso desde una muestra sesgada — pero la literatura muestra que el costo del "
             "sesgo crece con el tiempo por concept drift y feedback loop (Kim et al., IEEE TKDE 2023; Mai "
             "et al., ICDMW 2021), algo que una simulación de una ronda subestima por construcción. El valor "
             "inmediato e insustituible de la banda aleatoria es otro: es el único estimador insesgado de la "
             f"tasa poblacional ({r['tasa_banda_aleatoria']:.1%} en la banda, contra "
             f"{r['tasa_obs_heuristica']:.1%} 'observado' en la muestra sesgada — casi 3 veces más) y el "
             "termómetro para detectar drift. El Protocolo de la Res. 56/2026 ya habilita controles "
             "aleatorios: el diseño institucional necesario existe."),
            ("Limitaciones",
             "(1) Datos sintéticos: las relaciones se construyeron según la estructura del régimen argentino "
             "y la evidencia internacional, pero las magnitudes reales pueden diferir; los resultados validan "
             "arquitectura y metodología, no rendimiento productivo. (2) La correlación precio-incumplimiento "
             "se simuló moderada porque carece de evidencia publicada a nivel de firma: es una hipótesis que "
             "el propio sistema permitirá medir por primera vez. (3) El fraude se desplaza entre canales "
             "cuando se enforza uno (Yang 2008): el modelo requiere reentrenamiento periódico y vigilancia "
             "de drift. (4) No se simuló el canal courier/uso personal, que la evidencia de EE.UU. señala "
             "como vía de fuga; merece un módulo propio (destinatarios con envíos repetidos de la misma "
             "especie y vendedores sin importaciones comerciales registradas)."),
            ("Próximos pasos",
             "(1) Reemplazar el generador por los datos reales: despachos SIM con respuestas RT, DJC públicas "
             "vía QR, certificados de los organismos, RENAI, alertas internacionales. (2) Desplegar la Etapa "
             "1 como tablero de inconsistencias (las reglas son verificables hoy y cada hallazgo ya tiene "
             "tipificación sancionatoria). (3) Acordar la banda aleatoria (1–2%) dentro del Protocolo de la "
             "Res. 56/2026 y registrar resultados de ensayo de forma estructurada. (4) Tras 12–24 meses de "
             "etiquetas, entrenar la Etapa 2 con validación temporal, explicabilidad por caso (SHAP) para "
             "sustento administrativo, y métrica de gestión precision@k según capacidad real de los "
             "laboratorios. (5) Medir y publicar la correlación subvaluación-incumplimiento técnico: vacío "
             "explícito en la literatura internacional."),
        ], pie="Prueba de concepto sobre datos sintéticos · Repositorio: Reglamentación Técnica · junio 2026")
