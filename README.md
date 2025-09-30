# Datathon Decision — IA para Shortlist de Candidatos

**Autor:** Matheus Amaral  
**Curso:** Pós-Tech FIAP — DTAT — Fase 5  
**Data:** 2025-09-30

## 1) Problema
Priorizar candidatos por vaga (shortlist) para acelerar a triagem e melhorar taxa de acerto do time de recrutamento.

## 2) Dados e ETL
- 3 fontes JSON: `vagas`, `prospects`, `applicants`.
- Expansão de estruturas aninhadas; limpeza de datas/moedas/booleanos.
- Join → **consolidado** por candidatura (candidato × vaga) + label de contratação.

## 3) Validação
Split **temporal 80/20** (sem vazamento).

## 4) Modelo
Pipeline **One-Hot Encoder + RandomForest (balanced)** em scikit-learn.

### Métricas (teste)
- **AUC ROC:** 0.8678  
- **PR-AUC:** 0.2578  
- **F1 @ 0.50:** 0.3019  
- **Precisão @ 0.50:** 0.2009  
- **Recall @ 0.50:** 0.6074

## 5) Estratégia de decisão (para operação)
**Top-N por vaga** com corte mínimo de probabilidade.

- **N:** 2  
- **p_min:** 0.45

**Métricas dessa política (teste):**
- Precisão: 0.2922
- Recall: 0.5493
- F1: 0.3815
- Hit@N: 0.3958

## 6) Como reproduzir (notebook)
1. Executar ETL + consolidação.  
2. Treinar o pipeline (OHE + RF).  
3. Gerar probabilidades no teste.  
4. Aplicar política **Top-2** com **p_min=0.45**.  
5. Exportar: `shortlist_top2_p045.csv`.

## 7) Artefatos
- **Modelo:** `models/modelo.joblib`  
- **Shortlist:** `shortlist_top2_p045.csv`  
- **Métricas:** `reports/metrics.json`

## 8) Próximos passos
- Features: *gap de senioridade*, sazonalidade, keywords técnicas.  
- Tuning leve focado em **PR-AUC**.  
- App **Streamlit** (upload CSV consolidado → probas → Top-N por vaga → download).

---

> Observação: ambiente Spark foi instável no Colab; fallback scikit-learn garantiu entrega com métricas consistentes.
