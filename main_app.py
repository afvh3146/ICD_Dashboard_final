import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -------------------------
# Configuración
# -------------------------
st.set_page_config(page_title="EDA - Streamlit", layout="wide")
st.title("📊 EDA en Streamlit (3 pestañas: Cuantitativo, Cualitativo y Gráfico)")
st.caption("Carga un CSV, define el tamaño de muestra y explora el dataset con análisis dinámico y controles.")

# -------------------------
# Sidebar: carga de datos
# -------------------------
st.sidebar.header("1) Carga de datos")

use_local = st.sidebar.checkbox("Usar archivo local energia_renovable.csv (si existe)", value=True)
uploaded = st.sidebar.file_uploader("O sube un archivo CSV", type=["csv"])

df = None
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.sidebar.success("CSV cargado desde el uploader")
elif use_local:
    try:
        df = pd.read_csv("energia_renovable.csv")
        st.sidebar.success("Cargado desde energia_renovable.csv")
    except Exception:
        df = None

if df is None:
    st.info("👈 Sube un CSV o agrega energia_renovable.csv al repositorio para iniciar.")
    st.stop()

# -------------------------
# Sidebar: opciones generales
# -------------------------
st.sidebar.header("2) Opciones generales")
auto_parse_dates = st.sidebar.checkbox("Intentar convertir columnas tipo fecha", value=True)

if auto_parse_dates:
    for col in df.columns:
        if df[col].dtype == "object":
            sample = df[col].dropna().astype(str).head(25)
            if len(sample) > 0 and (sample.str.contains(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", regex=True).mean() > 0.5):
                df[col] = pd.to_datetime(df[col], errors="coerce")

# -------------------------
# Sidebar: tamaño de muestra (submuestreo)
# -------------------------
st.sidebar.header("3) Tamaño de muestra (muestras a analizar)")
n_total = len(df)
default_n = min(500, n_total)

sample_n = st.sidebar.slider(
    "Número de filas a analizar",
    min_value=50 if n_total >= 50 else 1,
    max_value=n_total,
    value=default_n,
    step=10 if n_total >= 200 else 1,
)
seed = st.sidebar.number_input("Semilla (para muestreo reproducible)", min_value=0, max_value=999999, value=42, step=1)
do_sample = st.sidebar.checkbox("Aplicar muestreo aleatorio", value=(sample_n < n_total))

if do_sample and sample_n < n_total:
    df_work = df.sample(n=sample_n, random_state=int(seed)).reset_index(drop=True)
else:
    df_work = df.copy()

st.sidebar.caption(f"Trabajando con: {len(df_work)} filas (de {n_total}).")

# -------------------------
# Tipos de columnas (en df_work)
# -------------------------
num_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_work.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
date_cols = df_work.select_dtypes(include=["datetime64[ns]"]).columns.tolist()

# -------------------------
# Sidebar: toggles de secciones
# -------------------------
st.sidebar.header("4) Activar / desactivar secciones")

show_head = st.sidebar.checkbox("Mostrar muestra (head)", value=True)
show_dtypes = st.sidebar.checkbox("Mostrar tipos de datos", value=False)
show_missing = st.sidebar.checkbox("Mostrar análisis de faltantes", value=True)

enable_desc = st.sidebar.checkbox("Cuantitativo: estadística descriptiva", value=True)
enable_corr = st.sidebar.checkbox("Cuantitativo: correlaciones", value=True)
enable_top_corr = st.sidebar.checkbox("Cuantitativo: top correlaciones", value=True)
enable_outliers = st.sidebar.checkbox("Cuantitativo: outliers (IQR)", value=False)

enable_crosstab = st.sidebar.checkbox("Cualitativo: cruce de categóricas (crosstab)", value=False)
enable_group_summary = st.sidebar.checkbox("Cualitativo: resumen por grupo", value=True)

enable_hist_box = st.sidebar.checkbox("Gráfico: histograma + boxplot", value=True)
enable_box_by_cat = st.sidebar.checkbox("Gráfico: boxplot por categoría", value=True)
enable_scatter = st.sidebar.checkbox("Gráfico: scatter", value=True)
enable_ts = st.sidebar.checkbox("Gráfico: serie temporal (si hay fecha)", value=False)

# -------------------------
# Filtros globales (aplican a TODO, sobre df_work)
# -------------------------
st.sidebar.header("5) Filtros globales (opcional)")
filtered_df = df_work.copy()

# Filtro numérico
if len(num_cols) > 0:
    num_filter_col = st.sidebar.selectbox("Filtrar por numérica", ["(ninguna)"] + num_cols)
    if num_filter_col != "(ninguna)":
        minv = float(np.nanmin(filtered_df[num_filter_col]))
        maxv = float(np.nanmax(filtered_df[num_filter_col]))
        r = st.sidebar.slider(
            f"Rango: {num_filter_col}",
            min_value=minv,
            max_value=maxv,
            value=(minv, maxv),
        )
        filtered_df = filtered_df[(filtered_df[num_filter_col] >= r[0]) & (filtered_df[num_filter_col] <= r[1])]

# Filtro categórico
if len(cat_cols) > 0:
    cat_filter_col = st.sidebar.selectbox("Filtrar por categórica", ["(ninguna)"] + cat_cols)
    if cat_filter_col != "(ninguna)":
        options = sorted(filtered_df[cat_filter_col].dropna().unique().tolist())
        chosen = st.sidebar.multiselect("Valores", options, default=options[: min(3, len(options))])
        if chosen:
            filtered_df = filtered_df[filtered_df[cat_filter_col].isin(chosen)]

# Filtro por fecha
if len(date_cols) > 0:
    date_filter_col = st.sidebar.selectbox("Filtrar por fecha", ["(ninguna)"] + date_cols)
    if date_filter_col != "(ninguna)":
        dmin = pd.to_datetime(filtered_df[date_filter_col].min())
        dmax = pd.to_datetime(filtered_df[date_filter_col].max())
        if pd.notna(dmin) and pd.notna(dmax):
            dr = st.sidebar.date_input(
                "Rango de fechas",
                value=(dmin.date(), dmax.date()),
                min_value=dmin.date(),
                max_value=dmax.date(),
            )
            if isinstance(dr, tuple) and len(dr) == 2:
                start, end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
                filtered_df = filtered_df[(filtered_df[date_filter_col] >= start) & (filtered_df[date_filter_col] <= end)]

st.sidebar.caption(f"Después de filtros: {filtered_df.shape[0]} filas / {filtered_df.shape[1]} columnas")

# Recalcular tipos sobre filtered_df
num_cols_f = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_f = filtered_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
date_cols_f = filtered_df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()

# -------------------------
# Vista general
# -------------------------
st.subheader("📌 Vista general (dataset en análisis)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Filas analizadas", f"{filtered_df.shape[0]:,}".replace(",", "."))
c2.metric("Columnas", filtered_df.shape[1])
c3.metric("Numéricas", len(num_cols_f))
c4.metric("Categóricas", len(cat_cols_f))

if show_head:
    with st.expander("Ver muestra (head)"):
        st.dataframe(filtered_df.head(20), use_container_width=True)

if show_dtypes:
    with st.expander("Tipos de datos"):
        st.dataframe(pd.DataFrame({"columna": filtered_df.columns, "dtype": filtered_df.dtypes.astype(str)}), use_container_width=True)

if show_missing:
    missing = filtered_df.isna().sum().sort_values(ascending=False)
    missing_pct = (missing / len(filtered_df) * 100).round(2)
    missing_table = pd.DataFrame({"faltantes": missing, "faltantes_%": missing_pct})
    missing_table = missing_table[missing_table["faltantes"] > 0]

    with st.expander("Calidad de datos: valores faltantes"):
        if missing_table.empty:
            st.success("✅ No se encontraron valores faltantes (NaN).")
        else:
            st.dataframe(missing_table, use_container_width=True)
            fig_miss = px.bar(
                missing_table.reset_index().rename(columns={"index": "columna"}),
                x="columna",
                y="faltantes",
                title="Faltantes por columna",
            )
            st.plotly_chart(fig_miss, use_container_width=True)

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3 = st.tabs(["1) Cuantitativo", "2) Cualitativo", "3) Gráfico"])

# =========================
# TAB 1: CUANTITATIVO
# =========================
with tab1:
    st.header("1) EDA Cuantitativo (numéricas)")

    if len(num_cols_f) == 0:
        st.warning("No se detectaron columnas numéricas en el dataset filtrado.")
    else:
        if enable_desc:
            st.subheader("Estadística descriptiva + faltantes")
            desc = filtered_df[num_cols_f].describe().T
            desc["missing"] = filtered_df[num_cols_f].isna().sum()
            desc["missing_%"] = (desc["missing"] / len(filtered_df) * 100).round(2)
            st.dataframe(desc, use_container_width=True)

        if enable_corr:
            st.subheader("Correlaciones (numéricas)")
            if len(num_cols_f) >= 2:
                method = st.selectbox("Método de correlación", ["pearson", "spearman"], key="corr_method")
                corr = filtered_df[num_cols_f].corr(method=method)
                fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title=f"Matriz de correlación ({method})")
                st.plotly_chart(fig_corr, use_container_width=True)

                if enable_top_corr:
                    st.subheader("Top relaciones (|corr| más altas)")
                    corr_abs = corr.abs()
                    upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))
                    top_pairs = (
                        upper.stack()
                        .sort_values(ascending=False)
                        .head(10)
                        .reset_index()
                        .rename(columns={"level_0": "Variable 1", "level_1": "Variable 2", 0: "|corr|"})
                    )
                    st.dataframe(top_pairs, use_container_width=True)
            else:
                st.info("Se requieren al menos 2 columnas numéricas para correlación.")

        if enable_outliers:
            st.subheader("Outliers (IQR) — columnas seleccionadas")
            cols_for_outliers = st.multiselect(
                "Selecciona columnas numéricas para detectar outliers",
                num_cols_f,
                default=num_cols_f[: min(3, len(num_cols_f))],
                key="out_cols",
            )
            if cols_for_outliers:
                out_summary = []
                for col in cols_for_outliers:
                    s = filtered_df[col].dropna()
                    if len(s) == 0:
                        continue
                    q1, q3 = np.percentile(s, 25), np.percentile(s, 75)
                    iqr = q3 - q1
                    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    out_count = int(((s < low) | (s > high)).sum())
                    out_summary.append([col, float(q1), float(np.median(s)), float(q3), float(low), float(high), out_count])

                out_df = pd.DataFrame(
                    out_summary,
                    columns=["columna", "Q1", "Mediana", "Q3", "Límite inferior", "Límite superior", "Nº outliers"],
                )
                st.dataframe(out_df, use_container_width=True)

# =========================
# TAB 2: CUALITATIVO
# =========================
with tab2:
    st.header("2) EDA Cualitativo (categóricas)")

    if len(cat_cols_f) == 0:
        st.info("No se detectaron columnas categóricas (texto/bool) en el dataset filtrado.")
    else:
        st.subheader("Frecuencias (Top N) + gráfico")
        cat = st.selectbox("Variable categórica", cat_cols_f, key="cat_main")
        top_n = st.slider("Top N categorías", 5, 50, 15, key="topn_cat")

        freq = filtered_df[cat].astype("string").fillna("NaN").value_counts().head(top_n).reset_index()
        freq.columns = [cat, "conteo"]

        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(freq, use_container_width=True)
        with c2:
            fig_bar = px.bar(freq, x=cat, y="conteo", title=f"Top {top_n} categorías en {cat}")
            st.plotly_chart(fig_bar, use_container_width=True)

        if enable_crosstab and len(cat_cols_f) >= 2:
            st.subheader("Cruce entre dos categóricas (tabla de contingencia)")
            cat2 = st.selectbox("Segunda categórica", [c for c in cat_cols_f if c != cat], key="cat_second")
            ct = pd.crosstab(filtered_df[cat].fillna("NaN"), filtered_df[cat2].fillna("NaN"))
            st.dataframe(ct, use_container_width=True)

            fig_ct = px.imshow(ct, text_auto=True, aspect="auto", title=f"Mapa de calor: {cat} vs {cat2}")
            st.plotly_chart(fig_ct, use_container_width=True)

        if enable_group_summary and len(num_cols_f) > 0:
            st.subheader("Resumen por grupo (categórica → numéricas)")
            group_cat = st.selectbox("Agrupar por", cat_cols_f, key="group_cat")
            target_num = st.selectbox("Métrica numérica", num_cols_f, key="group_num")
            agg = st.selectbox("Agregación", ["mean", "median", "sum", "min", "max", "count"], key="group_agg")

            grp = (
                filtered_df.groupby(group_cat)[target_num]
                .agg(agg)
                .sort_values(ascending=False)
                .reset_index()
                .rename(columns={target_num: f"{agg}({target_num})"})
            )
            st.dataframe(grp.head(30), use_container_width=True)

            fig_grp = px.bar(grp.head(30), x=group_cat, y=f"{agg}({target_num})", title=f"{agg} de {target_num} por {group_cat}")
            st.plotly_chart(fig_grp, use_container_width=True)

# =========================
# TAB 3: GRÁFICO
# =========================
with tab3:
    st.header("3) EDA Gráfico (distribuciones, boxplots, relaciones)")

    if len(num_cols_f) == 0:
        st.warning("No hay columnas numéricas para graficar con los filtros actuales.")
    else:
        if enable_hist_box:
            st.subheader("Histograma + Boxplot (variable numérica)")
            col = st.selectbox("Columna numérica", num_cols_f, key="plot_num")
            bins = st.slider("Bins del histograma", 5, 60, 20, 1, key="bins_hist")

            c1, c2 = st.columns(2)
            with c1:
                fig_hist = px.histogram(filtered_df, x=col, nbins=bins, title=f"Histograma: {col}")
                st.plotly_chart(fig_hist, use_container_width=True)
            with c2:
                fig_box = px.box(filtered_df, y=col, points="outliers", title=f"Boxplot: {col}")
                st.plotly_chart(fig_box, use_container_width=True)

        if enable_box_by_cat and len(cat_cols_f) > 0:
            st.subheader("Boxplot por categoría (comparación de grupos)")
            cat = st.selectbox("Categórica para agrupar", cat_cols_f, key="plot_cat")
            col2 = st.selectbox("Variable numérica", num_cols_f, key="plot_num2")
            fig_box_cat = px.box(filtered_df, x=cat, y=col2, points="outliers", title=f"{col2} por {cat}")
            st.plotly_chart(fig_box_cat, use_container_width=True)

        if enable_scatter and len(num_cols_f) >= 2:
            st.subheader("Scatter (relación entre dos numéricas)")
            x = st.selectbox("Eje X", num_cols_f, index=0, key="scatter_x")
            y = st.selectbox("Eje Y", num_cols_f, index=1 if len(num_cols_f) > 1 else 0, key="scatter_y")

            color_opt = ["(sin color)"] + cat_cols_f
            color_by = st.selectbox("Color por (opcional)", color_opt, key="scatter_color")

            # Trendline OLS opcional (no rompe si falta statsmodels)
            add_trendline = st.checkbox("Agregar línea de tendencia (OLS)", value=False, key="trend_toggle")
            trendline_arg = None
            if add_trendline:
                try:
                    import statsmodels.api as sm  # noqa: F401
                    if filtered_df[[x, y]].dropna().shape[0] >= 3:
                        trendline_arg = "ols"
                    else:
                        st.info("No hay suficientes datos (mín. 3 puntos) para calcular la tendencia.")
                except ModuleNotFoundError:
                    st.warning("Para usar OLS debes instalar 'statsmodels'. Se mostrará el scatter sin tendencia.")
                    trendline_arg = None

            fig_sc = px.scatter(
                filtered_df,
                x=x,
                y=y,
                color=None if color_by == "(sin color)" else color_by,
                trendline=trendline_arg,
                title=f"Scatter: {y} vs {x}",
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        if enable_ts and len(date_cols_f) > 0:
            st.subheader("Serie temporal (si hay columna fecha)")
            date_col = st.selectbox("Columna fecha", date_cols_f, key="date_col")
            y_ts = st.selectbox("Variable numérica", num_cols_f, key="y_ts")

            tmp = filtered_df[[date_col, y_ts]].dropna().sort_values(date_col)
            if tmp.empty:
                st.info("No hay datos suficientes para la serie temporal con los filtros actuales.")
            else:
                freq = st.selectbox("Frecuencia", ["D (diario)", "W (semanal)", "M (mensual)"], key="ts_freq")
                rule = {"D (diario)": "D", "W (semanal)": "W", "M (mensual)": "M"}[freq]

                ts = tmp.set_index(date_col).resample(rule)[y_ts].mean().reset_index()
                fig_ts = px.line(ts, x=date_col, y=y_ts, title=f"Serie temporal ({freq}) de {y_ts}")
                st.plotly_chart(fig_ts, use_container_width=True)

# ======================================================================
# ASISTENTE DE ANÁLISIS (GROQ) - Sidebar
# ======================================================================
st.sidebar.markdown("---")
st.sidebar.header("🤖 Asistente de análisis (Groq + Llama 3.3)")

st.sidebar.caption(
    "Pega tu API key de Groq para hacer preguntas sobre el dataset filtrado. "
    "La key se usa solo en esta sesión."
)

api_key = st.sidebar.text_input("GROQ_API_KEY", type="password", help="No la subas al repo. Solo pégala aquí.")

# Opciones del asistente
with st.sidebar.expander("Opciones del asistente", expanded=False):
    model_name = st.selectbox(
        "Modelo",
        ["llama-3.3-70b-versatile"],
        index=0,
        help="Modelo recomendado para uso general.",
    )
    temperature = st.slider("Creatividad (temperature)", 0.0, 1.0, 0.3, 0.05)
    max_tokens = st.slider("Máx tokens de respuesta", 128, 2048, 700, 64)

def build_dataset_context(df_ctx: pd.DataFrame, num_cols_ctx, cat_cols_ctx) -> str:
    """Construye un contexto compacto del dataset para enviarlo al LLM."""
    # Faltantes
    miss = df_ctx.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)

    # Describe numérico (compacto)
    if len(num_cols_ctx) > 0:
        desc = df_ctx[num_cols_ctx].describe().T
        desc["missing"] = df_ctx[num_cols_ctx].isna().sum()
        desc = desc[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "missing"]]
        desc_txt = desc.round(4).to_string()
    else:
        desc_txt = "No hay columnas numéricas."

    # Frecuencias categóricas (top 5 por columna)
    cat_summaries = []
    for c in cat_cols_ctx[:10]:  # limita para no inflar contexto
        vc = df_ctx[c].astype("string").fillna("NaN").value_counts().head(5)
        cat_summaries.append(f"\n- {c} (top 5):\n{vc.to_string()}")
    cat_txt = "\n".join(cat_summaries) if cat_summaries else "No hay columnas categóricas."

    head_txt = df_ctx.head(10).to_string(index=False)

    miss_txt = miss.to_string() if not miss.empty else "Sin faltantes."

    ctx = f"""
DATASET CONTEXT (resumen)
- Filas (filtrado): {df_ctx.shape[0]}
- Columnas: {df_ctx.shape[1]}
- Columnas numéricas: {num_cols_ctx}
- Columnas categóricas: {cat_cols_ctx}

FALTANTES (columna -> cantidad):
{miss_txt}

DESCRIPTIVE STATS (numéricas):
{desc_txt}

CATEGÓRICAS (top 5 por columna, primeras 10 columnas categóricas):
{cat_txt}

MUESTRA (primeras 10 filas):
{head_txt}
""".strip()
    return ctx

# Memoria del chat
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": "Hola. Soy tu asistente de análisis. Pregúntame sobre patrones, outliers, correlaciones, ideas de gráficos, hipótesis o conclusiones del dataset.",
        }
    ]

# Render del chat en sidebar
with st.sidebar.expander("Chat", expanded=True):
    for m in st.session_state.chat_messages:
        if m["role"] == "user":
            st.markdown(f"**Tú:** {m['content']}")
        else:
            st.markdown(f"**Asistente:** {m['content']}")

    user_prompt = st.text_area(
        "Tu pregunta",
        placeholder="Ej: Resume los hallazgos clave. ¿Qué variables parecen más importantes? ¿Qué gráficos recomiendas?",
        height=80,
    )

    col_a, col_b = st.columns(2)
    send = col_a.button("Enviar", use_container_width=True, disabled=(not api_key or not user_prompt.strip()))
    clear = col_b.button("Limpiar chat", use_container_width=True)

    if clear:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": "Chat reiniciado. Pregúntame lo que quieras sobre el dataset filtrado.",
            }
        ]
        st.rerun()

    if send:
        try:
            from groq import Groq  # SDK oficial de Groq :contentReference[oaicite:2]{index=2}

            client = Groq(api_key=api_key)

            # Contexto del dataset filtrado (para que el modelo responda con base en datos)
            context = build_dataset_context(filtered_df, num_cols_f, cat_cols_f)

            system_msg = (
                "Eres un analista de datos experto. Responde con claridad, en español, "
                "sin inventar datos: usa SOLO el contexto del dataset que te doy. "
                "Si falta información, dilo y sugiere qué gráfico/consulta harías. "
                "Entrega respuestas accionables (insights, hipótesis, recomendaciones de EDA)."
            )

            # Guardar mensaje del usuario
            st.session_state.chat_messages.append({"role": "user", "content": user_prompt.strip()})

            # Preparar mensajes para Groq Chat Completions
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "system", "content": context},
            ]

            # Mantener algo de historial (últimos 6)
            for m in st.session_state.chat_messages[-6:]:
                messages.append({"role": m["role"], "content": m["content"]})

            resp = client.chat.completions.create(
                model=model_name,  # llama-3.3-70b-versatile :contentReference[oaicite:3]{index=3}
                messages=messages,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
            )

            answer = resp.choices[0].message.content
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})

            st.rerun()

        except Exception as e:
            st.error(f"Error al llamar Groq: {e}")

st.success("✅ App lista: muestreo + checkboxes + 3 pestañas + asistente Groq.")
