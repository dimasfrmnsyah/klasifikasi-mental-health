from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import LabelEncoder

# Unduh stopwords jika belum ada
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

app = Flask(__name__)

# Baca dataset untuk memperoleh label unik
df = pd.read_csv("Translated_Combined_Data.csv")
df = df.drop(columns=df.columns[0], axis=1)  # Drop index column jika ada

# Encoding label kategorikal
label_encoder = LabelEncoder()
df['encoded_status'] = label_encoder.fit_transform(df['status'])

# Load model dan TF-IDF Vectorizer
svm_model = joblib.load('svm_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Fungsi untuk preprocessing teks
def preprocess_text(text):
    if isinstance(text, str):  # Pastikan text adalah string
        text = text.lower()  # Convert ke lowercase
        text = re.sub(r'\d+', '', text)  # Hapus angka
        text = re.sub(r'[^\w\s]', '', text)  # Hapus tanda baca
        text = ' '.join([word for word in text.split() if word not in stop_words])  # Hapus stopwords
    else:
        text = ''  # Jika bukan string (misalnya NaN atau float), ganti dengan string kosong
    return text

@app.route('/', methods=['POST'])
def predict():
    data = request.get_json()

    if 'text' not in data:
        return jsonify({'error': 'Data tidak valid, kolom "text" harus ada!'}), 400

    text = data['text']

    cleaned_text = preprocess_text(text)

    X_input = vectorizer.transform([cleaned_text]).toarray()
    prediction = svm_model.predict(X_input)
    
    predicted_label_index = prediction[0]
    
    predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]

    prediction_numeric = int(predicted_label_index) + 1
    
    return jsonify({'prediction': prediction_numeric, 'label': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
