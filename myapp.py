import streamlit as st
import pandas as pd
from transformers import pipeline
from io import BytesIO

# ----------------- Load Models -----------------
@st.cache_resource
def load_models():
    return {
        "English": pipeline("sentiment-analysis",
                            model="tahamueed23/sentiment_roberta_english_finetuned"),
        "Urdu": pipeline("sentiment-analysis",
                         model="tahamueed23/fine_tuned_cardiffnlp_urdu_and_roman-urdu"),
        "Roman Urdu": pipeline("sentiment-analysis",
                               model="tahamueed23/fine_tuned_cardiffnlp_urdu_and_roman-urdu")
    }

models = load_models()

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="Pro Sentiment Analyzer", layout="wide")

# ----------------- CSS -----------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

/* ======== PROFESSIONAL BACKGROUND ======== */
body {
    background: linear-gradient(135deg, #e3edf7 0%, #f8fafc 100%);
}

/* ======== CENTERED CONTAINER ======== */
.main-container {
    max-width: 900px;
    margin: auto;
}

/* ======== HEADER BAR ======== */
.header {
    text-align: center;
    padding: 30px 10px;
    border-radius: 20px;
    background: linear-gradient(120deg, #6a85b6, #bac8e0);
    color: white;
    animation: fadeIn 1.2s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ======== GLASS CARD ======== */
.card {
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(15px);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(200,200,200,0.35);
    box-shadow: 0px 10px 30px rgba(0,0,0,0.08);
    transition: 0.3s ease;
}

.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 15px 35px rgba(0,0,0,0.12);
}

/* ======== BUTTON ======== */
.stButton>button {
    width: 100%;
    padding: 12px;
    font-size: 17px;
    background: #6a85b6;
    color: white;
    border-radius: 10px;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    background: #5774a3;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.15);
    transform: translateY(-2px);
}

</style>
""", unsafe_allow_html=True)

# ----------------- LAYOUT -----------------
st.markdown("<div class='main-container'><h3>Write sentences in English, Urdu and Roman Urdu</h3>", unsafe_allow_html=True)

st.markdown("<div class='header'><h1>Multilingual Sentiment Analyzer</h1></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<div class='card'><h6>1. Hafiz Taha Mueed</h6> <h6>2. Ghulam Mohi Ud Din</h6>", unsafe_allow_html=True)

text_input = st.text_area("Enter Sentences (one per line):", height=180, placeholder="Type sentences here...")
language = st.selectbox("Select Language", ["English", "Urdu", "Roman Urdu"])

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Analyze Sentiment"):
    if not text_input.strip():
        st.error("Enter at least one sentence.")
    else:
        sentences = [s.strip() for s in text_input.split("\n") if s.strip()]
        results = []

        for s in sentences:
            out = models[language](s)[0]
            results.append({
                "sentence": s,
                "language": language,
                "sentiment": out["label"],
                "confidence": round(float(out["score"]), 3)
            })

        df = pd.DataFrame(results)

        st.markdown("<div class='card'><h4>Here are the results of your given sentences.</h4>", unsafe_allow_html=True)
        st.subheader("Sentiment Results")
        st.dataframe(df)
        st.markdown("</div>", unsafe_allow_html=True)

        buffer = BytesIO()
        df.to_excel(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            "Download Excel Report",
            buffer,
            file_name="sentiment_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

st.markdown("</div>", unsafe_allow_html=True)
