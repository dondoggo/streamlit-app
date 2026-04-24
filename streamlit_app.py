import streamlit as st
import pandas as pd
from transformers import pipeline, MarianMTModel, MarianTokenizer

st.set_page_config(
    page_title="Zestaw narzędzi NLP",
    page_icon="🧠",
    layout="centered",
)

st.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Pytorch_logo.png/320px-Pytorch_logo.png",
    width=120,
)

st.title("Zestaw narzędzi NLP")
st.caption("Aplikacja do analizy tekstu i tłumaczeń z wykorzystaniem modeli językowych")

st.success("Aplikacja działa poprawnie i jest gotowa do użycia.")

st.header("Czym jest ta aplikacja?")
st.write(
    "To narzędzie pozwala w prosty sposób przetestować dwa zastosowania sztucznej inteligencji: "
    "**analizę emocji** w tekście angielskim oraz **automatyczne tłumaczenie** z angielskiego na niemiecki. "
    "Modele działają lokalnie — żadne dane nie są wysyłane do zewnętrznych serwisów."
)

st.subheader("Jak korzystać?")
st.write("1. Wybierz interesującą Cię funkcję z listy poniżej.")
st.write("2. Wpisz tekst w języku angielskim.")
st.write("3. Poczekaj chwilę — przy pierwszym uruchomieniu model musi się załadować.")

st.subheader("Przykładowy kod Streamlit")
st.code("st.write('Witaj, świecie!')", language="python")

with st.echo():
    st.write("Przykład działania st.echo() — ten kod jest widoczny i wykonywany jednocześnie")

st.subheader("Przykładowy zbiór danych")
df = pd.read_csv("DSP_4.csv", sep=";")
st.dataframe(df)

st.header("Narzędzia NLP")

st.info(
    "Wybierz jedną z opcji poniżej, wpisz tekst i kliknij poza polem tekstowym. "
    "Pierwsze uruchomienie może potrwać kilkadziesiąt sekund — model pobiera się automatycznie."
)

option = st.selectbox(
    "Wybierz funkcję",
    [
        "Analiza wydźwięku emocjonalnego (angielski)",
        "Tłumaczenie angielski → niemiecki",
    ],
)

@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis")

@st.cache_resource
def load_translator():
    model_name = "Helsinki-NLP/opus-mt-en-de"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

if option == "Analiza wydźwięku emocjonalnego (angielski)":
    text = st.text_area("Wpisz tekst w języku angielskim", placeholder="np. I really love this product!")
    if text:
        with st.spinner("Trwa analiza emocji — proszę czekać..."):
            classifier = load_sentiment()
            answer = classifier(text)
        label = answer[0]["label"]
        score = answer[0]["score"]
        if label == "POSITIVE":
            st.success(f"Wydźwięk pozytywny (pewność: {score:.1%})")
        else:
            st.error(f"Wydźwięk negatywny (pewność: {score:.1%})")

elif option == "Tłumaczenie angielski → niemiecki":
    text = st.text_area("Wpisz tekst w języku angielskim", placeholder="np. The weather is beautiful today.")
    if text:
        with st.spinner("Trwa tłumaczenie — proszę czekać..."):
            tokenizer, model = load_translator()
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            translated = model.generate(**inputs)
            result = tokenizer.decode(translated[0], skip_special_tokens=True)
        st.success("Tłumaczenie zakończone pomyślnie!")
        st.write("**Wynik tłumaczenia:**")
        st.write(result)

st.divider()
st.caption("Numer indeksu: 12345")