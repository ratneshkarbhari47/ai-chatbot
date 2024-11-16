from flask import Flask, request, jsonify
import nltk
import pickle
import numpy as np
import random

import json


# Load trained model, vectorizer, and class names
model = pickle.load(open('chatbot_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

app = Flask(__name__)

# Tokenizer and Lemmatizer
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = nltk.WordNetLemmatizer()

def preprocess_input(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    return ' '.join(tokens)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    user_input = data.get('message', '')

    if not user_input:
        return jsonify({"response": "Please type something."})

    # Preprocess user input and transform using vectorizer
    processed_input = preprocess_input(user_input)
    input_vector = vectorizer.transform([processed_input])

    # Get model prediction
    predicted_class = model.predict(input_vector)[0]
    tag = classes[predicted_class]

    # Load intents to fetch a response
    with open('intents.json') as file:
        intents = json.load(file)

    # Find the correct response based on the tag
    for intent in intents['intents']:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return jsonify({"response": response})
    
    return jsonify({"response": "Sorry, I didn't understand that."})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
