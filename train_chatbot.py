import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer and other variables
lemmatizer = WordNetLemmatizer()
vectorizer = CountVectorizer()
classes = []
documents = []
corpus = []
labels = []

# Load training data
with open('intents.json') as file:
    data = json.load(file)

# Prepare data for training
for intent in data['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        documents.append((word_list, intent['tag']))
        corpus.append(' '.join(word_list))
        labels.append(intent['tag'])
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

# Preprocess text
X = vectorizer.fit_transform(corpus)
y = np.array([classes.index(tag) for _, tag in documents])

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X, y)

# Save the model and vectorizer
pickle.dump(model, open('chatbot_model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))
print("Model trained and saved!")
