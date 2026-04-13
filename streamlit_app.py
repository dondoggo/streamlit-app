import streamlit as st
import pandas as pd
from transformers import pipeline

# ── Konfiguracja strony ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Toolkit – Lab05",
    page_icon="🌐",
    layout="centered",
)

st.success("Gratulacje! Z powodzeniem uruchomiłeś aplikację")

st.title("🌐 Lab05. Streamlit – NLP Toolkit")
st.caption("Aplikacja do analizy tekstu i tłumaczeń | Hugging Face Transformers")

st.header("Wprowadzenie do zajęć")
st.subheader("O Streamlit")
st.text("To przykładowa aplikacja z wykorzystaniem Streamlit")
st.write(
    "Streamlit jest biblioteką pozwalającą na uruchomienie modeli uczenia maszynowego."
)
st.code("st.write()", language="python")

with st.echo():
    st.write("Echo")

# ── Wczytanie CSV (ścieżka względna – działa lokalnie i na HF Spaces) ────────
df = pd.read_csv("DSP_4.csv", sep=";")
st.dataframe(df)

# ── Sekcja NLP ───────────────────────────────────────────────────────────────
st.header("Przetwarzanie języka naturalnego")

st.info(
    "**Jak korzystać?**  \n"
    "Wybierz jedną z opcji poniżej, wpisz tekst i poczekaj chwilę "
    "na załadowanie modelu (pierwsze uruchomienie może potrwać kilkadziesiąt sekund)."
)

option = st.selectbox(
    "Opcje",
    [
        "Wydźwięk emocjonalny tekstu (eng)",
        "Tłumaczenie angielski → niemiecki",
    ],
)

# ── Cachowanie modeli – ładowane raz, nie przy każdym kliknięciu ─────────────
@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis")

@st.cache_resource
def load_translator():
    return pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de")

# ── Opcja 1: Analiza sentymentu ───────────────────────────────────────────────
if option == "Wydźwięk emocjonalny tekstu (eng)":
    text = st.text_area(label="Wpisz tekst w języku angielskim")
    if text:
        with st.spinner("Analizuję wydźwięk emocjonalny..."):
            classifier = load_sentiment()
            answer = classifier(text)
        label = answer[0]["label"]
        score = answer[0]["score"]
        if label == "POSITIVE":
            st.success(f"Wydźwięk: **{label}** (pewność: {score:.1%})")
        else:
            st.error(f"Wydźwięk: **{label}** (pewność: {score:.1%})")

# ── Opcja 2: Tłumaczenie EN → DE ─────────────────────────────────────────────
elif option == "Tłumaczenie angielski → niemiecki":
    text = st.text_area(label="Wpisz tekst w języku angielskim")
    if text:
        with st.spinner("Tłumaczę tekst... (pierwsze uruchomienie ładuje model ~300 MB)"):
            translator = load_translator()
            result = translator(text, max_length=512)
        st.success("Tłumaczenie gotowe!")
        st.write("**Wynik:**")
        st.write(result[0]["translation_text"])

# ── Stopka ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Numer indeksu: XXXXX")  # <-- wpisz swój numer indeksu