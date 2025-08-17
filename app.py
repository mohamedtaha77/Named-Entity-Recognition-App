# app.py
# -----------------------------
# NER (Named Entity Recognition) demo for news text
# - Rule-based baseline (EntityRuler)
# - spaCy small model: en_core_web_sm
# - spaCy transformer model: en_core_web_trf (optional)
# Features:
#   * Text box or .txt file upload
#   * displaCy visualization embedded in Streamlit
#   * Entities table + CSV download
#   * Label filter and simple stats
# -----------------------------

import io
import json
import pandas as pd
import streamlit as st

import spacy
from spacy import displacy
from spacy.pipeline import EntityRuler

# Try to import spacy-transformers (optional, for en_core_web_trf)
try:
    import spacy_transformers  # noqa: F401
    HAS_TRF = True
except Exception:
    HAS_TRF = False

# -----------------------------
# Page config & small CSS
# -----------------------------
st.set_page_config(page_title="NER — News Articles", layout="centered")
st.markdown(
    """
    <style>
      .small { color:#777; font-size:0.9rem; }
      .stTextArea textarea { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Named Entity Recognition (NER) - News Articles")
st.markdown(
    "Paste a paragraph or upload a `.txt` file. Choose a pipeline and extract entities "
    "(e.g., PERSON, ORG, LOC, MISC)."
)

# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource
def load_spacy_model(name: str):
    """Load a spaCy pipeline; download if missing."""
    try:
        return spacy.load(name)
    except OSError:
        from spacy.cli import download
        download(name)
        return spacy.load(name)

@st.cache_resource
def build_rule_nlp():
    """Very small rule-based baseline using EntityRuler."""
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler")
    patterns = [
        {"label": "ORG", "pattern": "United Nations"},
        {"label": "ORG", "pattern": "European Union"},
        {"label": "GPE", "pattern": "United States"},
        {"label": "GPE", "pattern": "Germany"},
        {"label": "PERSON", "pattern": [{"IS_TITLE": True}, {"IS_TITLE": True}]},  # e.g., "David Beckham"
        {"label": "MISC", "pattern": "Premier League"},
    ]
    ruler.add_patterns(patterns)
    return nlp

def ents_to_df(doc, allowed=None) -> pd.DataFrame:
    rows = []
    for e in doc.ents:
        if allowed and e.label_ not in allowed:
            continue
        rows.append({"text": e.text, "label": e.label_, "start": e.start_char, "end": e.end_char})
    return pd.DataFrame(rows)

def displacy_html(doc, allowed=None) -> str:
    if allowed:
        doc.set_ents([e for e in doc.ents if e.label_ in allowed])

    raw = displacy.render(
        doc, style="ent", page=False, jupyter=False, options={"compact": True}
    )

    return f"""
    <style>
      /* make unlabeled text white */
      .entities {{ color:#ffffff !important; }}
      /* keep entity text/labels dark for contrast (covers both 'entity' and 'ent' classes) */
      .entities mark.entity, .entities mark.entity * ,
      .entities mark.ent,    .entities mark.ent    * {{ color:#111111 !important; }}
    </style>
    {raw}
    """


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Settings")

model_choice = st.sidebar.radio(
    "Pipeline",
    (
        "spaCy small (en_core_web_sm)",
        "spaCy transformer (en_core_web_trf)",
        "Rule-based (EntityRuler)",
    ),
)

default_labels = ["PERSON", "ORG", "LOC", "GPE", "MISC"]
labels_filter = st.sidebar.multiselect(
    "Show only labels (optional)",
    default_labels
    + ["NORP", "FAC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE",
       "MONEY", "DATE", "TIME", "PERCENT", "QUANTITY", "ORDINAL", "CARDINAL"],
    default=[],  # no filter by default
)

st.sidebar.markdown(
    "<div class='small'>Tip: Transformer is more accurate but heavier. "
    "If hosting on Streamlit Cloud, first load can take longer.</div>",
    unsafe_allow_html=True,
)

# -----------------------------
# Input area
# -----------------------------
sample_text = (
    "Barack Obama spoke in Berlin about NATO and the European Union. "
    "Later, he returned to the United States to meet Microsoft executives."
)

uploaded = st.file_uploader("Or upload a .txt file", type=["txt"])
if uploaded is not None:
    try:
        text = uploaded.read().decode("utf-8")
    except Exception:
        text = io.TextIOWrapper(uploaded, encoding="utf-8").read()
else:
    text = st.text_area("Input text", value=sample_text, height=180)

# -----------------------------
# Load selected pipeline
# -----------------------------
if model_choice.startswith("spaCy small"):
    nlp = load_spacy_model("en_core_web_sm")
elif model_choice.startswith("spaCy transformer"):
    if not HAS_TRF:
        st.warning(
            "`spacy-transformers` not available in this environment. "
            "Falling back to en_core_web_sm."
        )
        nlp = load_spacy_model("en_core_web_sm")
    else:
        nlp = load_spacy_model("en_core_web_trf")
else:
    nlp = build_rule_nlp()

# -----------------------------
# Action
# -----------------------------
if st.button("Extract Entities"):
    if not text.strip():
        st.info("Please enter or upload some text.")
        st.stop()

    doc = nlp(text)

    # Table
    df = ents_to_df(doc, allowed=set(labels_filter) if labels_filter else None)
    st.subheader("Entities")
    if df.empty:
        st.write("No entities extracted (or filtered out).")
    else:
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="entities.csv",
            mime="text/csv",
        )

    # Visualization
    st.subheader("Visualization")
    html = displacy_html(doc, allowed=set(labels_filter) if labels_filter else None)
    st.components.v1.html(html, height=260, scrolling=True)

    # Quick stats
    st.subheader("Stats")
    by_label = df["label"].value_counts().rename_axis("label").reset_index(name="count") if not df.empty else pd.DataFrame(columns=["label","count"])
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total entities", int(by_label["count"].sum()) if not by_label.empty else 0)
    with col2:
        st.metric("Unique labels", by_label.shape[0])
    if not by_label.empty:
        st.bar_chart(by_label.set_index("label"))

    # Optional raw JSON
    with st.expander("Show JSON"):
        st.code(json.dumps(df.to_dict(orient="records"), indent=2), language="json")

# Footer
st.markdown("<div style='text-align: center;'>Made with ❤️ for Elevvo Internship Task 4</div>", unsafe_allow_html=True)
