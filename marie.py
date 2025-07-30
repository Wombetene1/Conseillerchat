import streamlit as st
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Chargement des fichiers sauvegardés
vectorizer = joblib.load("vectorizer.pkl")
X = joblib.load("X_matrix.pkl")
df = joblib.load("dialogue_df.pkl")

# Fonction chatbot
def chatbot(user_input):
    input_vec = vectorizer.transform([user_input])
    sim_scores = cosine_similarity(input_vec, X)
    idx = np.argmax(sim_scores)


    # Si l'index est dans la première moitié => client, sinon => conseiller
    if idx < len(df):
        return df['conseiller'].iloc[idx]
    else:
        return df['conseiller'].iloc[idx - len(df)]

# Interface utilisateur Streamlit
def main():
    st.title("💬 Chatbot Conseiller Financier")
    st.write("Posez une question sur les crédits, comptes, dépôts à terme, etc.")

    user_input = st.text_input("Votre question :", "")

    if user_input:
        response = chatbot(user_input)
        st.write("🤖 Réponse :", response)

if __name__ == "__main__":
    main()

