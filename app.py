from flask import Flask, request, jsonify, render_template
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import requests
import numpy as np
from keras.models import load_model
import json
import random

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
intents = json.loads(open('data1.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Add your ChatGPT API key and endpoint here
CHATGPT_API_KEY = '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'
CHATGPT_API_ENDPOINT = 'https://api.openai.com/v1/engines/gpt-3.5-turbo/completions'

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

def generate_chatgpt_response(user_message):
    try:
        params = {
            'prompt': user_message,
            'max_tokens': 50,
        }

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {CHATGPT_API_KEY}',
        }

        response = requests.post(CHATGPT_API_ENDPOINT, json=params, headers=headers)

        if response.status_code == 200:
            data = response.json()
            chatgpt_response = data['choices'][0]['text'].strip()
            return chatgpt_response
        else:
            return 'Error: Something went wrong with ChatGPT.'
    except Exception as e:
        print('Error:', e)
        return 'Error: Something went wrong with ChatGPT.'

@app.route("/")
def index():
    return render_template("index1.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/bot")
def bot():
    return render_template("chatbot.html")

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json['message']
        response = chatbot_response(user_message)
        chatgpt_response = generate_chatgpt_response(user_message)
        combined_response = f"Chatbot: {response}\nChatGPT: {chatgpt_response}"
        return jsonify({"message": combined_response})
    except Exception as e:
        return jsonify({"message": "Error: " + str(e)})

if __name__ == '__main__':
    app.run(debug=True)
