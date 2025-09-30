import io
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Shortlist de Candidatos", layout="wide")

@st.cache_resource
def load_model(path="models/modelo.joblib"):
    return joblib.load(path)

st.title("📋 Shortlist de Candidatos — Datathon")
st.caption("Regra default: Top-2 por vaga com p_min=0.45")

# 1) Carregar modelo
try:
    model = load_model()
except Exception as e:
    st.error(f"Erro ao carregar modelo: {e}")
    st.stop()

# Colunas padrão (caso não consiga ler do pipeline salvo)
DEFAULT_CAT = [
    "estado","modalidade","tipo_contratacao",
    "nivel_profissional_vaga","nivel_ingles_vaga","nivel_espanhol_vaga",
    "nivel_profissional_cand","nivel_ingles_cand","nivel_espanhol_cand",
]
DEFAULT_NUM = ["dias_ate_update"]

def get_expected_cols(model_obj):
    """Tenta descobrir as colunas esperadas pelo ColumnTransformer salvo"""
    try:
        pre = getattr(model_obj, "named_steps", {}).get("pre", None) or getattr(model_obj, "pre", None)
        trfs = getattr(pre, "transformers", None) or getattr(pre, "transformers_", None)
        cats, nums = [], []
        if trfs:
            for name, trans, cols in trfs:
                if name == "cat":
                    cats = list(cols)
                elif name == "num":
                    nums = list(cols)
        return (cats or DEFAULT_CAT), (nums or DEFAULT_NUM)
    except Exception:
        return DEFAULT_CAT, DEFAULT_NUM

CAT_COLS, NUM_COLS = get_expected_cols(model)

# 2) Upload do CSV consolidado
csv = st.file_uploader("Envie o CSV consolidado (mesmo schema usado no treino)", type=["csv"])

# 3) Controles
colA, colB, colC = st.columns([1,1,2])
with colA:
    N = st.slider("Top-N por vaga", 1, 10, 2, 1)
with colB:
    p_min = st.slider("Prob. mínima (p_min)", 0.00, 1.00, 0.45, 0.01)
with colC:
    st.write(" ")

def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Deixa o CSV 100% compatível com o pipeline salvo:
       - Nada de pd.NA (troca por '' em cat e por 0.0 em num)
       - Dtypes: cat -> object (string Python), num -> float
    """
    df = df.copy()

    # IDs como string simples (sem pd.NA)
    for c in ["id_vaga", "id_candidato"]:
        if c in df.columns:
            df[c] = df[c].astype(str).replace({"<NA>": "", "nan": "", "NaN": ""})

    # Categóricas: garantir existência, tirar pd.NA e forçar object
    for c in CAT_COLS:
        if c not in df.columns:
            df[c] = "None"
        df[c] = (
            df[c]
            .astype(object)                    # <- object, não "string" do pandas
            .where(pd.notna(df[c]), "None")    # pd.NA/NaN -> "None"
            .replace({"<NA>": "None", "": "None"})
        )

    # Numéricas: garantir existência, converter, e trocar NA/Inf por 0.0 (float)
    for c in NUM_COLS:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = (
            pd.to_numeric(df[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )

    # Por via das dúvidas, remove QUALQUER pd.NA remanescente no DF inteiro
    df = df.replace({pd.NA: np.nan})  # fora das colunas de entrada não afeta
    return df

if csv is not None:
    df_raw = pd.read_csv(csv)
    st.subheader("Amostra do arquivo")
    st.dataframe(df_raw.head(10), use_container_width=True)

    # 4) Sanidade básica: precisa ter id_vaga e id_candidato
    need_cols = {"id_vaga", "id_candidato"}
    if not need_cols.issubset(set(df_raw.columns)):
        st.error(f"O CSV precisa conter as colunas: {sorted(list(need_cols))}.")
        st.stop()

    # 5) Saneamento para casar com o pipeline salvo
    df = sanitize_dataframe(df_raw)

    # 6) Scorar usando exatamente as colunas que o modelo espera
    X = df[CAT_COLS + NUM_COLS].copy()
    try:
        proba = model.predict_proba(X)[:, 1]
    except Exception as e:
        st.error(
            "Falha ao pontuar. O CSV precisa ter o mesmo schema de treino.\n\n"
            f"Detalhe: {e}\n\n"
            f"Esperado (cat): {CAT_COLS}\nEsperado (num): {NUM_COLS}"
        )
        st.stop()

    out = df[["id_vaga", "id_candidato"]].copy()
    out["proba"] = proba

    # 7) Regra Top-N + p_min
    filtrado = out[out["proba"] >= p_min]
    topn = (
        filtrado.sort_values(["id_vaga", "proba"], ascending=[True, False])
        .groupby("id_vaga", as_index=False)
        .head(N)
    )

    st.subheader("Shortlist gerada")
    st.write(f"Vagas cobertas: {topn['id_vaga'].nunique()} · Linhas: {len(topn)}")
    st.dataframe(
        topn.sort_values(["id_vaga", "proba"], ascending=[True, False]),
        use_container_width=True,
    )

    # 8) Download
    csv_bytes = topn.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Baixar shortlist (CSV)",
        data=csv_bytes,
        file_name=f"shortlist_top{N}_p{p_min:.2f}.csv",
        mime="text/csv",
    )

    # 9) (Opcional) métricas se houver y_true no CSV
    if "y_true" in df.columns:
        from sklearn.metrics import precision_score, recall_score, f1_score
        pred_s = pd.Series(0, index=out.index, dtype=int)
        pred_s.loc[topn.index] = 1
        try:
            prec = precision_score(df["y_true"], pred_s)
            rec  = recall_score(df["y_true"], pred_s)
            f1   = f1_score(df["y_true"], pred_s)
            st.markdown(f"**Precisão:** {prec:.3f} · **Recall:** {rec:.3f} · **F1:** {f1:.3f}")
        except Exception:
            pass
else:
    st.info("Envie um CSV consolidado para gerar a shortlist.")
