# 🐦 Twitter Bot Detection Using Machine Learning and NLP

### Comparing Structured Data, Engineered Text Features, and Pure NLP Approaches

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-brightgreen.svg)
![CatBoost](https://img.shields.io/badge/CatBoost-Gradient%20Boosting-yellow.svg)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple.svg)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

## 📌 Project Overview

Twitter (X) bots — automated or semi-automated accounts — undermine platform integrity by inflating
engagement metrics, distorting trending topics, spreading disinformation, and distributing spam or
phishing links. This project builds and rigorously compares machine learning models that classify
Twitter accounts as **human** or **bot**, with a specific focus on answering one question:

> **How much does an account's bio (`description`) actually contribute to bot detection — and can bio text alone do the job?**

The project runs **three independent, head-to-head experiments** to isolate the answer, then explains
the final model with feature importance and SHAP, and ships it as a deployable Streamlit app.

## 📊 Dataset Description

**File:** `twitter_human_bots_dataset.csv`
**Target variable:** `account_type` → `human` | `bot`

| Column | Description |
|---|---|
| `created_at` | Account creation timestamp |
| `default_profile` | Whether the account uses the default profile theme |
| `default_profile_image` | Whether the account uses the default (non-custom) profile photo |
| `description` | Account bio / free text |
| `favourites_count` | Number of tweets liked by the account |
| `followers_count` | Number of followers |
| `friends_count` | Number of accounts followed |
| `geo_enabled` | Whether geo-tagging is enabled |
| `lang` | Account language code |
| `location` | Free-text profile location |
| `statuses_count` | Total number of tweets posted |
| `verified` | Whether the account is verified |
| `average_tweets_per_day` | Average posting frequency |
| `account_age_days` | Account age in days |
| `account_type` | **Target** — `human` or `bot` |

*(`Unnamed: 0`, `id`, `profile_background_image_url`, `profile_image_url`, and `screen_name` are dropped
during cleaning — see Methodology.)*

## 🧪 Methodology

1. **Data Cleaning** — drop identifier/leakage columns, engineer `has_location` from `location`, impute
   missing `description`/`lang`, remove duplicates, normalize boolean flags.
2. **Exploratory Data Analysis** — distributional and behavioral comparison of human vs. bot accounts
   across all structured features (histograms, boxplots, countplots, correlation heatmap).
3. **Three Experiments** (see below).
4. **Hyperparameter Tuning** — `RandomizedSearchCV` on the best Experiment 2 model.
5. **Explainability** — global feature importance + SHAP summary/bar plots.
6. **Deployment** — artifacts saved with `joblib`; a complete Streamlit app for interactive predictions.

## 🔬 The Three Experiments

| Experiment | Feature Set | Models Trained |
|---|---|---|
| **1 — Structured Only** | Account metadata only, **no** `description` | Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost |
| **2 — Structured + Description Features** | Experiment 1 features **+** 9 engineered bio features (`bio_length`, `bio_word_count`, `has_description`, `mention_count`, `url_count`, `has_url`, `uppercase_ratio`, `digit_count`, `special_character_count`) | Same 5 models |
| **3 — NLP (Description Only)** | `description` text only, vectorized with **TF-IDF (max 500 features)** | Logistic Regression, Linear SVM, Multinomial Naive Bayes |

Every model in every experiment is evaluated on the same five metrics for a fair, apples-to-apples
comparison: **Accuracy, Precision, Recall, F1 Score, ROC AUC**.

## 🏆 Model Comparison & Results

Each experiment produces a leaderboard sorted by F1 Score (`baseline_results`, `engineered_feature_results`,
`nlp_results`), and Section 12 of the notebook combines the best model from each into one final comparison
table and chart, directly answering:

1. Does the description improve performance?
2. Are engineered text features useful?
3. Can bio text alone detect bots?
4. Which overall approach performs best?

*(Exact metric values depend on the dataset run — the notebook is fully reproducible and regenerates this
table on every execution.)*

The final, deployed pipeline (used for tuning, feature importance, SHAP, and the Streamlit app) is the
**Experiment 2 — Structured + Description Features** model, which offers the strongest balance of
predictive performance and interpretability.

## 🛠 Technologies Used

- **Data handling:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly, WordCloud
- **Machine Learning:** scikit-learn, XGBoost, LightGBM, CatBoost
- **NLP:** NLTK, TF-IDF (scikit-learn)
- **Explainability:** SHAP
- **Deployment:** Streamlit, joblib
- **Environment:** Google Colab / Jupyter

## ⚙️ Installation

```bash
git clone <your-repo-url>
cd twitter-bot-detection
pip install -r requirements.txt
```

**`requirements.txt`:**
```
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
xgboost
lightgbm
catboost
shap
nltk
wordcloud
streamlit
joblib
```

## 🚀 Usage

### Run the notebook
1. Open `Twitter_Bot_Detection_ML_NLP.ipynb` in Google Colab.
2. Run all cells top to bottom — when prompted, upload `twitter_human_bots_dataset.csv`.
3. Trained model artifacts are saved to `./artifacts/` (Section 16) and zipped for download.

### Run the Streamlit app
```bash
# After running the notebook through Section 16 (artifacts saved):
streamlit run app.py
```
Then open the local URL Streamlit prints in your browser, fill in the account features and bio text in
the sidebar/form, and click **Predict** to see the Human/Bot classification, bot probability, and the
top contributing factors for that prediction.

## 🔮 Future Improvements

- Incorporate network-level / graph-based features (follower-following graph structure, coordinated
  posting timing) to catch coordinated botnets.
- Replace TF-IDF with transformer-based embeddings (e.g., fine-tuned DistilBERT) for richer, multilingual
  bio representations.
- Add temporal/time-series modeling of posting behavior (time-of-day patterns, burstiness).
- Calibrate the deployed decision threshold to the cost asymmetry between false positives and false
  negatives for a specific platform's moderation policy.
- Periodically retrain and monitor for feature-importance drift, since bot operators actively adapt to
  evade detection.



---

## 👤 Author

 Data Scientist | ML Engineer  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/mariam-kedr-mariamahmed)
---
