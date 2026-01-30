import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="EDA - Streamlit", layout="wide")
st.title("📊 EDA en Streamlit (3 pestañas: Cuantitativo, Cualitativo y Gráfico)")
st.caption("Carga un CSV, aplica filtros globales y explora el dataset con análisis dinámico.")

# -------------------------
# Sidebar: carga + opciones
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

st.sidebar.header("2) Opciones")
auto_parse_dates = st.sidebar.checkbox("Intentar convertir columnas tipo fecha", value=True)

if auto_parse_dates:
    for col in df.columns:
        if df[col].dtype == "object":
            sample = df[col].dropna().astype(str).head(25)
            if len(sample) > 0 and (sample.str.contains(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", regex=True).mean() > 0.5):
                df[col] = pd.to_datetime(df[col], errors="coerce")

# Tipos
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
date_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()

# -------------------------
# Vista general + tipos + faltantes
# -------------------------
st.subheader("📌 Vista general")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Filas", f"{df.shape[0]:,}".replace(",", "."))
c2.metric("Columnas", df.shape[1])
c3.metric("Numéricas", len(num_cols))
c4.metric("Categóricas", len(cat_cols))

with st.expander("Ver muestra (head)"):
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("Tipos de datos"):
    st.dataframe(pd.DataFrame({"columna": df.columns, "dtype": df.dtypes.astype(str)}), use_container_width=True)

# Faltantes
missing = df.isna().sum().sort_values(ascending=False)
missing_pct = (missing / len(df) * 100).round(2)
missing_table = pd.DataFrame({"faltantes": missing, "faltantes_%": missing_pct})
missing_table = missing_table[missing_table["faltantes"] > 0]

with st.expander("Calidad de datos: faltantes"):
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
# Filtros globales (aplican a TODO)
# -------------------------
st.sidebar.header("3) Filtros globales (opcional)")
filtered_df = df.copy()

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

# Filtro por fecha (si existe)
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
                filtered_df = filtered_df[
                    (filtered_df[date_filter_col] >= start) & (filtered_df[date_filter_col] <= end)
                ]

st.sidebar.caption(f"Filtrado: {filtered_df.shape[0]} filas / {filtered_df.shape[1]} columnas")

# -------------------------
# Pestañas del EDA
# -------------------------
tab1, tab2, tab3 = st.tabs(["1) Cuantitativo", "2) Cualitativo", "3) Gráfico"])

# =========================
# TAB 1: CUANTITATIVO
# =========================
with tab1:
    st.header("1) EDA Cuantitativo (numéricas)")
    if len(num_cols) == 0:
        st.warning("No se detectaron columnas numéricas.")
    else:
        st.subheader("Estadística descriptiva + faltantes")
        desc = filtered_df[num_cols].describe().T
        desc["missing"] = filtered_df[num_cols].isna().sum()
        desc["missing_%"] = (desc["missing"] / len(filtered_df) * 100).round(2)
        st.dataframe(desc, use_container_width=True)

        st.subheader("Correlaciones (numéricas)")
        if len(num_cols) >= 2:
            method = st.selectbox("Método de correlación", ["pearson", "spearman"], key="corr_method")
            corr = filtered_df[num_cols].corr(method=method)

            fig_corr = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                title=f"Matriz de correlación ({method})",
            )
            st.plotly_chart(fig_corr, use_container_width=True)

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

        st.subheader("Outliers (IQR) — columnas seleccionadas")
        cols_for_outliers = st.multiselect(
            "Selecciona columnas numéricas para detectar outliers",
            num_cols,
            default=num_cols[: min(3, len(num_cols))],
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
    if len(cat_cols) == 0:
        st.info("No se detectaron columnas categóricas (texto/bool).")
    else:
        st.subheader("Frecuencias (Top N) + gráfico")
        cat = st.selectbox("Variable categórica", cat_cols, key="cat_main")
        top_n = st.slider("Top N categorías", 5, 50, 15, key="topn_cat")

        freq = filtered_df[cat].astype("string").fillna("NaN").value_counts().head(top_n).reset_index()
        freq.columns = [cat, "conteo"]

        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(freq, use_container_width=True)
        with c2:
            fig_bar = px.bar(freq, x=cat, y="conteo", title=f"Top {top_n} categorías en {cat}")
            st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Cruce entre dos categóricas (tabla de contingencia)")
        if len(cat_cols) >= 2:
            cat2 = st.selectbox("Segunda categórica", [c for c in cat_cols if c != cat], key="cat_second")
            ct = pd.crosstab(filtered_df[cat].fillna("NaN"), filtered_df[cat2].fillna("NaN"))
            st.dataframe(ct, use_container_width=True)

            fig_ct = px.imshow(ct, text_auto=True, aspect="auto", title=f"Mapa de calor: {cat} vs {cat2}")
            st.plotly_chart(fig_ct, use_container_width=True)

        st.subheader("Resumen por grupo (categórica → numéricas)")
        if len(num_cols) > 0:
            group_cat = st.selectbox("Agrupar por", cat_cols, key="group_cat")
            target_num = st.selectbox("Métrica numérica", num_cols, key="group_num")
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
    if len(num_cols) == 0:
        st.warning("No hay columnas numéricas para graficar.")
    else:
        st.subheader("Histograma + Boxplot (variable numérica)")
        col = st.selectbox("Columna numérica", num_cols, key="plot_num")
        bins = st.slider("Bins del histograma", 5, 60, 20, 1, key="bins_hist")

        c1, c2 = st.columns(2)
        with c1:
            fig_hist = px.histogram(filtered_df, x=col, nbins=bins, title=f"Histograma: {col}")
            st.plotly_chart(fig_hist, use_container_width=True)
        with c2:
            fig_box = px.box(filtered_df, y=col, points="outliers", title=f"Boxplot: {col}")
            st.plotly_chart(fig_box, use_container_width=True)

        st.subheader("Boxplot por categoría (comparación de grupos)")
        if len(cat_cols) > 0:
            cat = st.selectbox("Categórica para agrupar", cat_cols, key="plot_cat")
            fig_box_cat = px.box(filtered_df, x=cat, y=col, points="outliers", title=f"{col} por {cat}")
            st.plotly_chart(fig_box_cat, use_container_width=True)

        st.subheader("Scatter (relación entre dos numéricas)")
        if len(num_cols) >= 2:
            x = st.selectbox("Eje X", num_cols, index=0, key="scatter_x")
            y = st.selectbox("Eje Y", num_cols, index=1 if len(num_cols) > 1 else 0, key="scatter_y")

            color_opt = ["(sin color)"] + cat_cols
            color_by = st.selectbox("Color por (opcional)", color_opt, key="scatter_color")

            fig_sc = px.scatter(
                filtered_df,
                x=x,
                y=y,
                color=None if color_by == "(sin color)" else color_by,
                trendline="ols" if filtered_df[[x, y]].dropna().shape[0] >= 3 else None,
                title=f"Scatter: {y} vs {x}",
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        st.subheader("Serie temporal (si hay columna fecha)")
        if len(date_cols) > 0:
            date_col = st.selectbox("Columna fecha", date_cols, key="date_col")
            y_ts = st.selectbox("Variable numérica", num_cols, key="y_ts")

            # ordenar y agrupar por fecha (día)
            tmp = filtered_df[[date_col, y_ts]].dropna().sort_values(date_col)
            if tmp.empty:
                st.info("No hay datos suficientes para la serie temporal con los filtros actuales.")
            else:
                freq = st.selectbox("Frecuencia", ["D (diario)", "W (semanal)", "M (mensual)"], key="ts_freq")
                rule = {"D (diario)": "D", "W (semanal)": "W", "M (mensual)": "M"}[freq]

                ts = tmp.set_index(date_col).resample(rule)[y_ts].mean().reset_index()
                fig_ts = px.line(ts, x=date_col, y=y_ts, title=f"Serie temporal ({freq}) de {y_ts}")
                st.plotly_chart(fig_ts, use_container_width=True)

st.success("✅ App lista: EDA en 3 pestañas con filtros globales y visualizaciones dinámicas.")
