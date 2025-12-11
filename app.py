import streamlit as st
import pandas as pd
from transformers import pipeline
from io import BytesIO

# ----------------- Load Models -----------------
@st.cache_resource
def load_models():
    models = {
        "English": pipeline("sentiment-analysis",
                            model="siebert/sentiment-roberta-large-english"),
        "Urdu": pipeline("sentiment-analysis",
                         model="mrgmd01/SA_Model_bert-base-multilingual-uncased"),
        "Roman Urdu": pipeline("sentiment-analysis",
                               model="mrgmd01/SA_Model_bert-base-multilingual-uncased")
    }
    return models

models = load_models()

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Sentence Sentiment Analyzer", layout="wide")
st.title("Multilingual Sentence Sentiment Analyzer")

st.write("Enter multiple sentences (one per line) and select the language for analysis.")

# User input
text_input = st.text_area("Enter sentences here:", height=200)
language = st.selectbox("Select language:", ["English", "Urdu", "Roman Urdu"])

if st.button("Analyze Sentences"):
    if not text_input.strip():
        st.error("Please enter at least one sentence.")
    else:
        sentences = [s.strip() for s in text_input.strip().split("\n") if s.strip()]
        results = []

        for sentence in sentences:
            output = models[language](sentence)[0]
            results.append({
                "sentence": sentence,
                "language": language,
                "sentiment": output["label"],
                "confidence": round(float(output["score"]), 3)
            })

        # Show results in Streamlit
        df = pd.DataFrame(results)
        st.subheader("Analysis Results")
        st.dataframe(df)

        # Save to Excel
        excel_buffer = BytesIO()
        df.to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)

        st.download_button(
            label="Download Results as Excel",
            data=excel_buffer,
            file_name="sentiment_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

st.markdown("---")
st.write("App ready. All sentences analyzed are saved into Excel with columns: sentence, language, sentiment, confidence.")
