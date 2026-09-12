import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st


# ---------------------------------------------------------------------------
DATA_PATH = "dataset/enhanced_student_habits_performance_dataset.csv"

st.set_page_config(
    page_title="Estudo x Redes Sociais x Desempenho",
    layout="wide",
)


@st.cache_data
def carregar_dados(caminho: str) -> pd.DataFrame:
    return pd.read_csv(caminho)


df = carregar_dados(DATA_PATH)

# ---------------------------------------------------------------------------

st.title("Estudo x Redes Sociais x Desempenho dos Alunos")
st.markdown(
    "Este dashboard investiga como o **tempo de estudo** e o **tempo em "
    "redes sociais** se relacionam com o **desempenho em exames**, usando "
    "o dataset *Student Habits and Academic Performance*."
)

st.divider()

# ---------------------------------------------------------------------------

st.sidebar.header("Filtros")

# Widget 1: selectbox de curso
cursos = ["Todos"] + sorted(df["major"].unique().tolist())
curso_selecionado = st.sidebar.selectbox("Curso (major)", cursos)

# Widget 2: slider de faixa de horas em redes sociais
min_rs, max_rs = float(df["social_media_hours"].min()), float(df["social_media_hours"].max())
faixa_rs = st.sidebar.slider(
    "Horas por dia em redes sociais",
    min_value=min_rs,
    max_value=max_rs,
    value=(min_rs, max_rs),
    step=0.5,
)

# Aplicando os filtros
df_filtrado = df[
    (df["social_media_hours"] >= faixa_rs[0]) & (df["social_media_hours"] <= faixa_rs[1])
]
if curso_selecionado != "Todos":
    df_filtrado = df_filtrado[df_filtrado["major"] == curso_selecionado]

st.sidebar.markdown(f"**{len(df_filtrado):,}** alunos após o filtro".replace(",", "."))

# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Nota média", f"{df_filtrado['exam_score'].mean():.1f}")
col2.metric("Horas de estudo/dia (média)", f"{df_filtrado['study_hours_per_day'].mean():.1f}")
col3.metric("Horas em redes sociais/dia (média)", f"{df_filtrado['social_media_hours'].mean():.1f}")

st.divider()

# ---------------------------------------------------------------------------
st.subheader("1. Horas de estudo x Nota do exame")
fig_dispersao = px.scatter(
    df_filtrado,
    x="study_hours_per_day",
    y="exam_score",
    color="social_media_hours",
    color_continuous_scale="RdYlGn_r",
    labels={
        "study_hours_per_day": "Horas de estudo por dia",
        "exam_score": "Nota do exame",
        "social_media_hours": "Horas em redes sociais",
    },
    opacity=0.6,
    trendline="ols",
)
st.plotly_chart(fig_dispersao, use_container_width=True)
st.caption(
    "Cada ponto é um aluno. A linha de tendência mostra a relação geral "
    "entre tempo de estudo e nota; a cor indica quanto tempo o aluno passa "
    "em redes sociais."
)

st.divider()

# ---------------------------------------------------------------------------

st.subheader("2. Nota do exame por faixa de uso de redes sociais")

bins = [-0.01, 1, 2, 3, 4, 100]
labels = ["até 1h", "1-2h", "2-3h", "3-4h", "mais de 4h"]
df_filtrado = df_filtrado.copy()
df_filtrado["faixa_redes_sociais"] = pd.cut(
    df_filtrado["social_media_hours"], bins=bins, labels=labels
)

fig_boxplot = px.box(
    df_filtrado,
    x="faixa_redes_sociais",
    y="exam_score",
    color="faixa_redes_sociais",
    category_orders={"faixa_redes_sociais": labels},
    points=False,
    labels={
        "faixa_redes_sociais": "Horas por dia em redes sociais",
        "exam_score": "Nota do exame",
    },
)
fig_boxplot.update_layout(showlegend=False)
st.plotly_chart(fig_boxplot, use_container_width=True)
st.caption(
    "Compara a distribuição das notas entre alunos que usam pouca e muita "
    "rede social por dia."
)

st.divider()

# --------------------------------------------------------------------------- 
st.subheader("3. Panorama geral: correlação entre hábitos e desempenho")

colunas_correlacao = [
    "study_hours_per_day", "social_media_hours", "netflix_hours",
    "sleep_hours", "attendance_percentage", "stress_level",
    "mental_health_rating", "motivation_level", "exam_score",
]
correlacao = df[colunas_correlacao].corr()

fig_heatmap, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(
    correlacao, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
    square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
)
ax.set_title("Correlação entre hábitos dos alunos e desempenho no exame", pad=12)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

st.pyplot(fig_heatmap)
st.caption(
    "Correlação de Pearson entre hábitos dos alunos e a nota final. "
    "Estudo e motivação têm relação positiva com a nota; redes sociais "
    "e Netflix praticamente não têm correlação linear com o desempenho "
    "nesse conjunto de dados."
)

st.divider()
st.caption("Dataset: Student Habits and Academic Performance (Kaggle) · Dashboard feito com Streamlit + Plotly express + Seaborn + Matplotlib ")