import streamlit as st
import pandas as pd
from joblib import load

st.title('Previsão do setor destinatário das reclamações da CAC utilizando processamento de linguagem natural.')
st.image('ai.jpg', width=300)
st.markdown('Digite o texto da reclamação')
reclamacao = st.text_area("Reclamação")

if st.button('Enviar'):
    st.write(reclamacao)

vect = load('arquivo_do_modelo/vectorizer.joblib')
clf = load('arquivo_do_modelo/compressed_ia_grct.joblib')

#Preparando texto para predição

import string
from sklearn.feature_extraction.text import CountVectorizer

def punctuation_removal(messy_str):
    clean_list = [char for char in messy_str if char not in string.punctuation]
    clean_str = ''.join(clean_list)
    return clean_str

texto = reclamacao

texto = texto.lower()

texto = punctuation_removal(texto)

texto = vect.transform([texto])

predicao = clf.predict(texto)[0]

if reclamacao:   
    st.markdown('O setor que receberá a reclamação: ' + predicao)




#Remover o menu da lateral
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
