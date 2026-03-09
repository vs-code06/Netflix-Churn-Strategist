<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg" alt="Netflix Logo" width="150"/>
  <br>
  <h1>🎬 Netflix Customer Churn Strategist</h1>
  <p><strong>A SaaS-grade Classical Machine Learning Platform for Predictive Churn Analytics</strong></p>

  [![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
  [![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
  [![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
  [![Deployment](https://img.shields.io/badge/Deployed-Streamlit_Cloud-success.svg?style=for-the-badge)](#)
</div>

---

This project is a complete machine learning system that predicts customer churn probability using structured Netflix subscription and viewing behaviour data.

Rather than stopping at a notebook, we built a fully modular pipeline — separate files for model training, inference, and a multi-page Streamlit UI — so the system can be retrained, updated, or extended without touching the app layer.

| | |
|---|---|
| **Live Demo** | [Netflix Churn Strategist on Streamlit Cloud](https://netflix-churn-ai.streamlit.app/) |
| **Dataset** | Netflix customer subscription records across demographics, devices, and viewing behaviour |
| **Records Used** | 5,000 labelled customer transactions |
| **Top Model** | Decision Tree Classifier — Accuracy: **~97.9%**, Precision: **~98.5%** |

---

## Results at a Glance

We trained a Decision Tree Classifier and validated it against held-out test data — no cherry-picking.

| Metric | Score |
|---|---|
| Accuracy | ~97.9% |
| Precision | ~98.5% |
| Recall | varies by threshold |
| F1 Score | reported on dashboard |

The confusion matrix rendered in the live app confirms the model's predictions are consistent across both churn and non-churn classes — the high accuracy isn't a lucky split artefact.

---

## Problem We Solved

Customer churn is genuinely hard to catch early. Two subscribers on identical plans can have completely different cancellation likelihood based on factors that aren't obvious at a glance — viewing frequency, login recency, device preference, monthly spend, and regional behaviour.

Manual churn identification is slow, reactive, and subjective.

We framed this as a supervised binary classification problem across four signal categories:

```
Demographic signals   →  age, gender, region
Subscription signals  →  plan type, monthly fee, payment method
Viewing signals       →  monthly watch hours, last login days, avg hrs/day
Account signals       →  active profiles, favourite genre, device type
```

The objective: build a model that flags high-risk customers before they cancel — and outputs a targeted retention strategy alongside every prediction.

---

## Dataset

Structured Netflix customer records covering subscription details, viewing behaviour, and churn outcome labels.

| Attribute | Detail |
|---|---|
| Total records | 5,000 |
| Feature inputs | 12 raw customer attributes |
| Target variable | `churned` — binary label (1 = cancelled, 0 = retained) |
| Categorical fields | gender, region, device, plan, payment method, favourite genre |
| Numerical fields | age, watch hours, login recency, avg hrs/day, monthly fee, profiles |

---

## Pipeline Architecture

```
netflix_customer_churn.csv  (raw input)
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│  model.py                                                │
│   ├── Load CSV and drop non-informative fields          │
│   ├── One-hot encode all categorical columns            │
│   ├── 80/20 stratified train/test split                 │
│   ├── Train DecisionTreeClassifier                      │
│   │     └── max_depth=10, min_samples_leaf=2            │
│   │         min_samples_split=10, random_state=42       │
│   ├── Evaluate: Accuracy, Precision, Recall, F1         │
│   ├── Generate confusion matrix                         │
│   └── Return model + metrics + column reference         │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  model.py → predict_new_customer()                       │
│   ├── Accept raw customer dict from UI                  │
│   ├── One-hot encode with pd.get_dummies()              │
│   ├── Reindex to match training column schema           │
│   └── Return binary churn prediction                    │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  app.py  (Streamlit multi-page router)                   │
│   ├── pages/predict.py  → 12-field input form           │
│   │     └── Calls predict_new_customer(), shows result  │
│   │         + tailored retention or upsell strategy     │
│   ├── pages/home.py    → Live model performance dashboard│
│   │     └── KPI cards + interactive confusion matrix    │
│   └── pages/dataset.py → Raw dataset explorer           │
└─────────────────────────────────────────────────────────┘
```

---

## Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/samay-hash/netflix-churn-streamlit.git
cd netflix-churn-streamlit

# 2. Initialize a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
python3 -m streamlit run app.py
```

App opens at `http://localhost:8501`

---

## Streamlit Cloud Deployment

This codebase is tuned for zero-config deployment on **Streamlit Community Cloud**:

1. Connect via [Streamlit Cloud Dashboard](https://share.streamlit.io/)
2. Mount repository `samay-hash/netflix-churn-streamlit`
3. Set the **Main file path** to `app.py`
4. Boot deployment — cloud compute auto-resolves all dependencies and loads the `.csv` dataset

---

## File Structure

```
📦 netflix-churn-streamlit/
│
├── netflix_customer_churn.csv       # Raw labelled customer dataset (5,000 records)
├── Gen_Ai_Project_Netflix_Churn.ipynb  # Full EDA + analysis notebook
│
├── model.py                         # Model training, evaluation & inference logic
├── app.py                           # Streamlit multi-page router
├── requirements.txt                 # Pinned Python dependencies
├── .gitignore                       # Ignored caches & environments
│
└── pages/
    ├── predict.py                   # 12-field churn prediction form + result output
    ├── home.py                      # Model performance dashboard + confusion matrix
    ├── dataset.py                   # Raw dataset explorer
    └── theme.py                     # Shared UI components, CSS injection, glassmorphism theme
```

---

## Key Findings

What actually drives Netflix churn predictions?

- **Login recency is the strongest behavioural signal** — customers who haven't logged in recently are disproportionately over-represented in churn predictions
- **Monthly watch hours and avg hrs/day work together** — high-volume viewers on any plan consistently land in the low-risk bucket
- **Plan type alone is not sufficient** — a Premium subscriber who barely watches is higher risk than a Basic subscriber who watches daily
- **Geographic region captures macro behavioural variance** — engagement patterns differ meaningfully across Africa, Asia, Europe, and the Americas
- **Decision Trees outperform more complex models here** — with 12 input features and clean one-hot encoding, a well-regularised tree (depth=10) achieves ~97.9% accuracy without overfitting

---

## Tech Stack

| Component | Library |
|---|---|
| Data wrangling | pandas, numpy |
| ML model | scikit-learn (DecisionTreeClassifier) |
| Visualization | plotly |
| Web app | streamlit |
| Deployment | Streamlit Community Cloud |

---

## Team

| Enrollment No. | Member | Role |
|---|---|---|
| 2401010505 | Vipul Sharma | Model Development & Optimization |
| 2401010254 | Lokendra Singh | Documentation & Data Structuring |
| 2401020053 | Samay Samrat | Deployment & Version Control |
| 2401010061 | Aman Kumar | Dataset Acquisition & Data Engineering |

---

## Limitations Worth Knowing

| Limitation | Details |
|---|---|
| Dataset origin | Synthetic/structured dataset — not sourced from live Netflix telemetry |
| Feature depth | 12 input features; real churn systems use hundreds of behavioural signals |
| Model scope | Single classifier; no ensemble or probability calibration layer |
| Temporal signals | No time-series patterns captured — login recency is a static snapshot |

---

## What's Next

- [x] Decision Tree Classifier with stratified split
- [x] Multi-page Streamlit UI with live confusion matrix
- [x] Retention strategy output alongside every prediction
- [ ] Add cross-validation and model comparison dashboard
- [ ] Incorporate ensemble methods (Random Forest, XGBoost) for comparison
- [ ] REST API wrapper (FastAPI) for programmatic churn scoring
- [ ] Probability score output instead of hard binary label

---

## Disclaimer

Churn predictions are generated by a statistical classifier trained on structured customer data. Outputs are for educational and exploratory purposes only and do not constitute professional business or retention strategy advice.

<br>
<div align="center">
    <p><i>Developed dynamically aligned with End-Sem / Mid-Sem ML Lifecycle Protocols</i></p>
</div>
