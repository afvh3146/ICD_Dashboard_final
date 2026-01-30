import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="EDA - Streamlit", layout="wide")

st.title("📊 EDA en Streamlit (Cuantitativo, Cualitativo y Gráfico)")
st.write("Carga un CSV y explora estadísticas, faltantes, distribuciones, boxplots y correlaciones.")

# -------------------------
# Carga de datos
# -------------------------
st.sidebar.header("1) Datos")

use_local = st.sidebar.checkbox("Usar archivo local energia_renovable.csv (si existe en el repo)", value=True)

df = None

if use_local:
    try:
        df = pd.read_csv("energia_renovable.csv")
        st.sidebar.success("Cargado desde energia_renovable.csv")
    except Exception as e:
        st.sidebar.warning("No se encontró energia_renovable.csv en el repo. Sube un CSV abajo.")
        df = None

uploaded = st.sidebar.file_uploader("O sube un archivo CSV", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.sidebar.success("CSV cargado desde el uploader")

if df is None:
    st.info("👈 Sube un CSV o agrega energia_renovable.csv al repositorio.")
    st.stop()

# -------------------------
# Limpieza mínima y tipos
# -------------------------
st.sidebar.header("2) Opciones")
parse_dates = st.sidebar.checkbox("Intentar convertir columnas tipo fecha automáticamente", value=True)

if parse_dates:
    # intenta convertir columnas que parezcan fecha
    for col in df.columns:
        if df[col].dtype == "object":
            sample = df[col].dropna().astype(str).head(20)
            # Heurística: si muchos tienen '-' o '/' o ':' o parecen fecha
            if sample.str.contains(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", regex=True).mean() > 0.5:
                df[col] = pd.to_datetime(df[col], errors="coerce")

# Separación de tipos
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
date_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()

# -------------------------
# Vista general
# -------------------------
st.subheader("📌 Vista general del dataset")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Filas", f"{df.shape[0]:,}".replace(",", "."))
c2.metric("Columnas", df.shape[1])
c3.metric("Numéricas", len(num_cols))
c4.metric("Categóricas", len(cat_cols))

with st.expander("Ver muestra de datos"):
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("Tipos de datos"):
    dtypes_df = pd.DataFrame({"columna": df.columns, "dtype": df.dtypes.astype(str).values})
    st.dataframe(dtypes_df, use_container_width=True)

# -------------------------
# 1) EDA Cuantitativo
# -------------------------
st.header("1) EDA Cuantitativo")

if len(num_cols) == 0:
    st.warning("No se detectaron columnas numéricas.")
else:
    st.subheader("Estadística descriptiva (numéricas)")
    desc = df[num_cols].describe().T
    desc["missing"] = df[num_cols].isna().sum()
    desc["missing_%"] = (desc["missing"] / len(df) * 100).round(2)
    st.dataframe(desc, use_container_width=True)

# -------------------------
# Datos faltantes
# -------------------------
st.header("2) Calidad de datos: valores faltantes")

missing = df.isna().sum().sort_values(ascending=False)
missing_pct = (missing / len(df) * 100).round(2)
missing_table = pd.DataFrame({"faltantes": missing, "faltantes_%": missing_pct})
missing_table = missing_table[missing_table["faltantes"] > 0]

if missing_table.empty:
    st.success("✅ No se encontraron valores faltantes (NaN).")
else:
    st.dataframe(missing_table, use_container_width=True)

    # Gráfico de faltantes
    fig_miss = px.bar(
        missing_table.reset_index().rename(columns={"index": "columna"}),
        x="columna",
        y="faltantes",
        title="Faltantes por columna",
    )
    st.plotly_chart(fig_miss, use_container_width=True)

# -------------------------
# 3) EDA Gráfico: distribuciones y boxplots
# -------------------------
st.header("3) EDA Gráfico")

if len(num_cols) > 0:
    st.subheader("Distribución (histograma) y caja y bigotes (boxplot)")
    col = st.selectbox("Selecciona una columna numérica", num_cols)

    c1, c2 = st.columns(2)

    with c1:
        bins = st.slider("Bins del histograma", min_value=5, max_value=60, value=20, step=1)
        fig_hist = px.histogram(df, x=col, nbins=bins, title=f"Histograma: {col}")
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        fig_box = px.box(df, y=col, points="outliers", title=f"Boxplot: {col}")
        st.plotly_chart(fig_box, use_container_width=True)

    # Boxplot por categoría (si existe una categórica)
    if len(cat_cols) > 0:
        st.subheader("Boxplot por categoría (comparación de grupos)")
        cat = st.selectbox("Selecciona una columna categórica para agrupar", cat_cols)

        fig_box_cat = px.box(df, x=cat, y=col, points="outliers", title=f"{col} por {cat}")
        st.plotly_chart(fig_box_cat, use_container_width=True)

# -------------------------
# 4) EDA Cualitativo
# -------------------------
st.header("4) EDA Cualitativo (categóricas)")

if len(cat_cols) == 0:
    st.info("No se detectaron columnas categóricas (texto).")
else:
    st.subheader("Frecuencias y distribución por categoría")
    cat = st.selectbox("Selecciona una variable categórica", cat_cols, key="cat_freq")
    top_n = st.slider("Top N categorías", 5, 50, 15)

    freq = df[cat].astype("string").fillna("NaN").value_counts().head(top_n).reset_index()
    freq.columns = [cat, "conteo"]

    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(freq, use_container_width=True)
    with c2:
        fig_bar = px.bar(freq, x=cat, y="conteo", title=f"Top {top_n} categorías en {cat}")
        st.plotly_chart(fig_bar, use_container_width=True)

# -------------------------
# 5) Correlación
# -------------------------
st.header("5) Correlaciones (numéricas)")

if len(num_cols) < 2:
    st.info("Se requieren al menos 2 columnas numéricas para correlación.")
else:
    method = st.selectbox("Método", ["pearson", "spearman"])
    corr = df[num_cols].corr(method=method)

    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title=f"Matriz de correlación ({method})",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# -------------------------
# 6) Segmentación simple / filtros (opcional)
# -------------------------
st.header("6) Filtros (opcional) para explorar subconjuntos")

if len(num_cols) > 0:
    num_filter_col = st.selectbox("Filtrar por columna numérica", ["(ninguna)"] + num_cols)
    filtered_df = df.copy()

    if num_filter_col != "(ninguna)":
        minv = float(np.nanmin(filtered_df[num_filter_col]))
        maxv = float(np.nanmax(filtered_df[num_filter_col]))
        r = st.slider(
            f"Rango para {num_filter_col}",
            min_value=minv,
            max_value=maxv,
            value=(minv, maxv),
        )
        filtered_df = filtered_df[(filtered_df[num_filter_col] >= r[0]) & (filtered_df[num_filter_col] <= r[1])]

    if len(cat_cols) > 0:
        cat_filter_col = st.selectbox("Filtrar por columna categórica", ["(ninguna)"] + cat_cols)
        if cat_filter_col != "(ninguna)":
            options = sorted(filtered_df[cat_filter_col].dropna().unique().tolist())
            chosen = st.multiselect(f"Valores para {cat_filter_col}", options, default=options[: min(3, len(options))])
            if chosen:
                filtered_df = filtered_df[filtered_df[cat_filter_col].isin(chosen)]

    st.write("Vista del dataset filtrado:")
    st.dataframe(filtered_df.head(50), use_container_width=True)
    st.caption(f"Filtrado: {filtered_df.shape[0]} filas / {filtered_df.shape[1]} columnas")

st.success("✅ EDA listo. Puedes añadir conclusiones y recomendaciones en la sección final del notebook o README.")

