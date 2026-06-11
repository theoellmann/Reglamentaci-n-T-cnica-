# Reglamentación Técnica
Para Caro C.

# Modelo de detección ex post de incumplimiento de reglamentos técnicos

Prueba de concepto de un modelo de riesgo para priorizar la fiscalización ex
post de productos importados alcanzados por reglamentos técnicos en Argentina,
bajo el régimen vigente 2024–2026 (Res. SIC 237/2024 — Marco General de
Evaluación de la Conformidad; Res. 16/2025, 313/2025 y sectoriales; Decreto
892/2025; Protocolo de Vigilancia de Mercado Res. 56/2026).

## Idea central

El régimen actual es declarativo y de control ex post: la Aduana no verifica
reglamentos técnicos en frontera; el importador declara su situación frente al
RT en el SIM (MALVINA), emite una Declaración Jurada de Conformidad (DJC) con
marcado QR, y la Secretaría de Industria y Comercio fiscaliza en el mercado.
El modelo prioriza qué productos comercializados mandar a verificación/ensayo,
maximizando la tasa de acierto con capacidad de inspección limitada
(métrica: precision@k).

**El target es técnico, no fiscal**: producto no conforme al reglamento
(riesgo de seguridad), no subvaluación.

## Arquitectura en dos etapas

1. **Etapa 1 — reglas deterministas (sin etiquetas)**: QR del marcado muerto,
   producto alertado en Safety Gate/CPSC/GlobalRecalls, certificado vencido o
   sobre-reutilizado, "uso idóneo" declarado para producto doméstico (inválido
   según Anexo V Res. 16/2025), precio unitario anómalo, reincidencia RENAI.
2. **Etapa 2 — gradient boosting supervisado**: entrenado con resultados de
   fiscalizaciones, manejando el sesgo de etiquetas selectivas (Lakkaraju et
   al. 2017) con una banda de inspecciones aleatorias (habilitada por la
   Res. 56/2026).

## Uso

```bash
pip install pandas numpy scikit-learn matplotlib
python3 run_pipeline.py
```

Genera en `outputs/`: `metricas.json` e `informe_modelo_rt.pdf` (informe
completo con metodología, resultados y limitaciones).

## Estructura

- `src/generar_datos.py` — generador de datos sintéticos del régimen argentino
  (importadores con arquetipos, respuestas RT del SIM, certificados, QR,
  alertas, canales de venta; incumplimiento latente + síntomas observables).
- `src/etapa1_reglas.py` — reglas deterministas y heurística histórica.
- `src/etapa2_modelo.py` — modelo supervisado, etiquetas selectivas,
  precision@k, importancias por permutación.
- `src/informe.py` — informe PDF.
- `run_pipeline.py` — orquestador.

## Advertencia

Los resultados se calculan sobre **datos sintéticos** construidos según la
estructura del régimen argentino y la evidencia internacional (línea
BACUDA/OMA — DATE KDD 2020; Vanhoeyveld et al. 2020; Kee & Nicita 2022;
campañas de vigilancia UE/UK). Validan la arquitectura y la metodología, no el
rendimiento sobre datos reales.
