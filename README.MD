# Prática de Random Forest (A Sabedoria da Floresta)

Este repositório contém uma aplicação prática e detalhada do algoritmo **Random Forest** aplicada ao dataset clínico **Wisconsin Breast Cancer**, comparando o desempenho de árvores individuais com florestas de decisão sob diversas configurações arquiteturais.

---

## 📊 Resumo Executivo das Métricas de Validação

Os modelos foram avaliados sob o rigor estatístico de um particionamento estratificado de **30% de teste** e controle total de reprodutibilidade via `random_state=42`.

| Arquitetura do Modelo | Acurácia (Treino) | Acurácia (Teste) | Gap de Generalização | Comportamento Observado |
| :--- | :---: | :---: | :---: | :--- |
| 🌳 **Árvore Solitária (Baseline)** | 100.00% | 91.81% | 8.19% | **Overfitting clássico:** Decora o treino e sofre com ruídos no teste. |
| 🪵 **Pequena Floresta (n=10)** | 100.00% | 92.98% | 7.02% | **Melhoria inicial:** A combinação de 10 opiniões reduz a variância. |
| 🌲 **Grande Floresta (n=150)** | 100.00% | 94.15% | 5.85% | **A Sabedoria da Multidão:** Máxima acurácia de teste. |
| ✂️ **Floresta Podada (n=150, depth=5)** | 99.25% | 94.15% | 5.09% | **O Estado da Arte (Recomendado):** Mínimo Gap e máxima estabilidade. |

---

## 💡 Insights e Análise Arquitetural

### 🛡️ O Trade-off Viés-Variância no Ambiente Clínico
No desenvolvimento de soluções de inteligência artificial aplicadas à saúde, **previsibilidade e estabilidade sobrepõem-se a picos marginais de acurácia**. 
* **Grande Floresta Livre (Modelo 3):** Apresenta 100% de acurácia no conjunto de treino. Ao não podar as árvores, permitimos que cada uma delas se aprofunde ao máximo, decorando ruídos e anomalias dos dados amostrados. Isso se reflete em um **Gap de Generalização de 5.85%**.
* **Floresta Podada (Modelo 4):** Ao limitar a profundidade em `max_depth=5`, introduzimos intencionalmente um leve viés no modelo (impedindo acurácia perfeita no treino), mas controlamos severamente a variância. O resultado é o menor **Gap de Generalização (5.09%)**, tornando este modelo o mais seguro e robusto para produção hospitalar.

### ⚡ Paralelização Massiva (`n_jobs=-1`)
A Random Forest destaca-se computacionalmente por ser um algoritmo **embarçadamente paralelo** (*embarrassingly parallel*). Isso se deve às propriedades da arquitetura de **Bagging**:
1. **Treinamento Assíncrono:** A geração da Árvore $N$ não depende, em nenhum momento, dos resultados da Árvore $N-1$ (diferente de Boosting/GBDTs, onde cada árvore subsequente é ajustada sequencialmente sobre os resíduos das anteriores).
2. **Independência Estatística:** Cada estimador individual é treinado com um subconjunto de amostras (*Bootstrap*) e recursos (*Feature Subspace Projection*) gerados de forma independente.

---

## 🛠️ Como Executar o Projeto

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

2. Execute o script de resolução e atualização:
   ```bash
   python scripts/solve_exercises.py
   ```

3. Explore o Jupyter Notebook interativo em `notebooks/live_coding_random_forest.ipynb` para visualizar a explicabilidade das características (Feature Importance) e a distribuição dos votos do ensemble para pacientes específicos.
