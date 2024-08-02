import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tkinter as tk
from tkinter import scrolledtext
import nltk
nltk.download('punkt')
nltk.download('stopwords')


# Load the dataset
data = pd.read_csv(r'C:\Users\ASUS\OneDrive\Desktop\python\dataset_cleaned.csv')

# Check if the dataset is loaded correctly
if data.empty:
    raise ValueError("Dataset is empty or not loaded correctly. Please check the file path and content.")

# Preprocess the dataset
stop_words = set(stopwords.words('english'))
data['question'] = data['question'].apply(lambda x: ' '.join([word for word in word_tokenize(x.lower()) if word.isalnum() and word not in stop_words]))

# Fit TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['question'])

def chatbot_response(user_query):
    user_query = ' '.join([word for word in word_tokenize(user_query.lower()) if word.isalnum() and word not in stop_words])
    user_vec = vectorizer.transform([user_query])
    similarities = cosine_similarity(user_vec, X)
    idx = np.argmax(similarities)
    return data.iloc[idx]['answer']

# GUI Application
def send():
    user_query = user_input.get()
    chat_area.insert(tk.END, "You: " + user_query + "\n")
    response = chatbot_response(user_query)
    chat_area.insert(tk.END, "Bot: " + response + "\n\n")
    user_input.delete(0, tk.END)

root = tk.Tk()
root.title("College Chatbot")

chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD)
chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

user_input = tk.Entry(root, width=100)
user_input.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
user_input.bind("<Return>", lambda event: send())

send_button = tk.Button(root, text="Send", command=send)
send_button.pack(padx=10, pady=10)

root.mainloop()
