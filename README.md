
# Datathon Decision — Shortlist Inteligente de Candidatos

**Autor:** Matheus Amaral  
**Curso:** Pós-Tech FIAP — DTAT — Fase 5  
**Data:** 2025-10-01

---

## 1) Contexto e objetivo

O desafio foi transformar dados históricos de **vagas** e **candidaturas** em um sistema que **prioriza candidatos** por vaga — gerando uma **shortlist** confiável, rápida de usar e fácil de explicar ao negócio.  
A proposta equilibra três perspectivas:
- **Engenharia de Dados:** pipeline reprodutível, limpeza robusta e união coerente das fontes.
- **Ciência de Dados/ML:** modelo supervisionado, avaliação honesta (split temporal) e foco em **precision–recall**.
- **Negócio:** regra clara de decisão (Top-N por vaga com corte mínimo de probabilidade) e um **app** simples para operação.

---

## 2) Dados & Engenharia (ETL)

**Fontes (JSON):** `vagas`, `prospects`, `applicants`.

**Principais passos**
- *Flatten* de estruturas aninhadas (dict/list) com preservação de chaves e semântica.
- Padronização de **datas** (DD-MM-AAAA), **moedas** (R$ → ponto decimal) e **booleanos** (sim/não/true/false/1/0).
- Higienização de textos (trim/casing) e normalização de categorias.
- **Join** por `id_vaga` e `id_candidato` → **consolidado** por candidatura *(linha = candidato × vaga)*.
- **Rótulo (label):** `y_contratado = 1` quando `situacao_candidato == "Contratado pela Decision"`.

**Features do modelo**
- Categóricas (vaga e candidato):  
  `estado`, `modalidade`, `tipo_contratacao`, `nivel_profissional_vaga`, `nivel_ingles_vaga`, `nivel_espanhol_vaga`,  
  `nivel_profissional_cand`, `nivel_ingles_cand`, `nivel_espanhol_cand`
- Numérica:  
  `dias_ate_update = ultima_atualizacao − data_candidatura` (em dias)

> O **split é temporal (80/20)** para refletir cenário real e evitar vazamento de informação.

---

## 3) Modelagem, validação e métricas

**Pipeline (scikit-learn):**
- `ColumnTransformer` com **OneHotEncoder** (categorias) e **SimpleImputer** (numéricos)
- **RandomForestClassifier** com `class_weight="balanced"` (classe positiva é rara)

**Por que Random Forest?**
- Robusto a categorias esparsas (via OHE) e a outliers moderados  
- Pouca sensibilidade a escala  
- Bom compromisso entre desempenho e interpretabilidade (importâncias)

### Resultados — *hold-out* (teste)
- **AUC-ROC:** 0.8932  
- **PR-AUC:** 0.7821  
- **F1 @ 0.50:** 0.6710  
- **Precisão @ 0.50:** 0.7023  
- **Recall @ 0.50:** 0.6421

> Métrica-guia: **PR-AUC** (mais informativa em desbalanceamento) + avaliação pela **política operacional** abaixo.

---

## 4) Política de decisão (operacional)

**Top-N por vaga** com **probabilidade mínima** (`p_min`).

- **N:** 5  
- **p_min:** 0.25

**Efeito da política no teste**
- **Precisão:** 0.8123  
- **Recall:** 0.5312  
- **F1:** 0.6412  
- **Hit@N:** 0.9451 *(% de vagas com ≥1 aprovado na shortlist)*

Como funciona:
1. O modelo estima **P(contratar | vaga, candidato)**.  
2. Filtra candidaturas com `prob >= p_min`.  
3. Ordena por probabilidade **dentro de cada vaga** e retorna os **Top-N**.

---

## 5) App (Streamlit) — experiência do usuário

- **Upload** de um CSV consolidado (mesmo schema usado no treino)  
- Ajuste de **Top-N** e **p_min** via *sliders*  
- Exibição da shortlist e **download** em CSV

**Schema mínimo esperado**
- IDs: `id_vaga`, `id_candidato`  
- Categóricas: `estado`, `modalidade`, `tipo_contratacao`, `nivel_profissional_vaga`, `nivel_ingles_vaga`, `nivel_espanhol_vaga`, `nivel_profissional_cand`, `nivel_ingles_cand`, `nivel_espanhol_cand`  
- Numérica: `dias_ate_update`

> O app trata ausências simples (preenchimentos *default*), mas a **fidelidade ao schema** eleva a qualidade da recomendação.

---

## 6) Reprodutibilidade e execução

**Notebook (ETL + treino):**
1. Executar ETL e gerar o **consolidado** por candidatura.  
2. Treinar o pipeline (OHE + RF) e salvar `models/modelo.joblib`.  
3. Exportar **shortlist** com a política padrão: **Top-5** e **p_min=0.25**.  
4. Registrar métricas em `reports/metrics.json`.

**App (local)**
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scriptsctivate
pip install -r requirements.txt
streamlit run app.py
```
