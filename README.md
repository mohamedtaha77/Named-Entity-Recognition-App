# 🧩 Named Entity Recognition (NER) on News — CoNLL-2003

A clean and interactive **NER app** built with **Streamlit**.  
It finds and labels entities in news text such as:

- 👤 **PERSON**
- 🏢 **ORG**
- 🗺️ **LOC / GPE**
- 🧩 **MISC**

Pipelines available in the UI:

- **Rule-based** baseline (spaCy **EntityRuler**)
- **spaCy small** (`en_core_web_sm`)
- **spaCy transformer** (`en_core_web_trf`)

---

## 📌 Features

✅ Paste text or upload a `.txt` file  
✅ Highlight entities with **displaCy** (inline)  
✅ Entities **table** with start/end offsets  
✅ **CSV download** of extracted entities  
✅ **Label filter** (show only PERSON/ORG/etc.)  
✅ Quick **stats** (entity counts)  
✅ Companion notebook with **strict span P/R/F1** evaluation on **CoNLL-2003**  
✅ Notebook cells kept ≤ **3 lines** (Task-3 style)

---

## 🔗 Live Demo (optional)

Deploy on Streamlit Cloud and put your link here:  
👉 `https://named-entity-recognition-app-77.streamlit.app/`

---

## 📁 Files Included

| File | Description |
|------|-------------|
| `app.py` | Streamlit UI for NER (rule-based + spaCy models) |
| `requirements.txt` | App dependencies |
| `Elevvo_NLP_Internship_Task4.ipynb` | End-to-end notebook: data parsing, evaluation, visualization |
| `train.txt`, `valid.txt`, `test.txt` | CoNLL-2003 (English) files (from Kaggle) |
| `ner_results_summary.csv` | Validation/Test **Precision/Recall/F1** for each pipeline |
| `ner_sample_predictions.csv` | 50 samples with **gold spans** vs **predicted spans** |
| `viz_sm.html`, `viz_trf.html`, `viz_rule.html` | Saved **displaCy** visualizations |

> You only need **`app.py` + `requirements.txt`** to *run the app*.  
> The dataset files are used in the notebook for evaluation.

---

## 🚀 How to Run Locally

1) **Clone the repo**
```bash
git clone https://github.com/yourname/ner-news-conll2003.git
cd ner-news-conll2003
```

2) **(Optional) Create a virtual environment**
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

3) **Install dependencies**
```bash
pip install -r requirements.txt
```

4) **Download spaCy models**
```bash
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf    # optional, heavier but stronger
```

5) **Run the app**
```bash
streamlit run app.py
```

---

## 🌍 Deployment (Streamlit Cloud)

- Go to https://streamlit.io/cloud  
- Connect your GitHub repo  
- Set **main file** to `app.py`  
- Add (optional) **Python version** and `requirements.txt` in the app settings  
- First load may take longer if you enable the transformer model

---

## 🧪 Notebook Workflow (what we did)

> `Elevvo_NLP_Internship_Task4.ipynb`

- **Data Collection**  
  - Download **CoNLL-2003 (English)** from Kaggle (`alaakhaled/conll003-englishversion`).  
  - Parse `train.txt`, `valid.txt`, `test.txt` into DataFrames with `tokens`, `ner_tags`, and `text`.

- **Preprocessing (NER-specific)**  
  - Keep **case and punctuation** (they are signals).  
  - Normalize whitespace only.  
  - Convert BIO tags to **gold character spans** for strict evaluation.

- **Pipelines**  
  - **Rule-based**: spaCy `EntityRuler` (few patterns for demo).  
  - **Model-based**: `en_core_web_sm` and `en_core_web_trf`.

- **Evaluation**  
  - Strict span-level **Precision / Recall / F1** (exact start, end, label must match).  
  - Results reported for **validation** and **test** splits.  
  - Per-label breakdown (PER/ORG/LOC/MISC) and quick label distributions (plots).

- **Visualization**  
  - **displaCy** highlighting; saved example HTMLs for each pipeline.

- **Outputs**  
  - `ner_results_summary.csv`, `ner_sample_predictions.csv`, and the three HTML visualizations.

**Key takeaways:**  
- Rule-based is precise on known patterns but misses a lot (low recall).  
- `en_core_web_sm` is a solid baseline.  
- `en_core_web_trf` generally **wins on F1** (better boundaries, fewer ORG↔GPE mix-ups).

---
