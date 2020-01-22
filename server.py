from flask import Flask
from flask import render_template
from flask import Flask, request, send_from_directory
from flask import jsonify

from transformers import *
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

import spacy
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

@app.route("/")
def inxex():
    return send_from_directory("templates", "index.html")

@app.route("/analyze", methods=["POST"])
def search():
    try:
        req = request.json
        text = req["text"]
        print(text)
        inputIds = torch.tensor(tokenizer.encode(text, add_special_tokens=False)).unsqueeze(0)  # Batch size 1
        print(inputIds)
        attention = model.forward(inputIds)[2]

        doc = nlp(text)

        words = list(map(lambda t: {'pos': t.pos_, 'text': t.text}, doc))
        attentions = list(map(lambda xxxx: xxxx, list(map(lambda xxx: xxx, list(map(lambda xx: xx, list(map(lambda x: x.tolist(), list(attention)))))))))
        data = {'words': words, 'attentions': attentions}
        
        return jsonify(data)
    except Exception as ex:
        return (jsonify(str(ex)), 500)
        

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8800)
