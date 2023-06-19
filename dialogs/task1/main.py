from typing import List, Dict

import numpy as np
import torch
from starlette.requests import Request

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from scipy.special import softmax
from scipy.stats import entropy
from ray import serve

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = DistilBertForSequenceClassification.from_pretrained(model_name)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

np.set_printoptions(suppress=True)

entropy_threshold = 0.693

mapper = ['negative', 'positive']
NEUTRAL_SENTIMENT = 'neutral'


def get_texts(json_data):
    # single text
    if isinstance(json_data, dict):
        return json_data["text"]

    if not isinstance(json_data, list):
        raise ValueError("Input must be a list or a dictionary.")

    # dataframe
    if isinstance(json_data[0], dict):
        return [d["text"] for d in json_data]

    if not isinstance(json_data[0], str):
        raise ValueError("Input must be a list of strings or a dictionary with a 'text' key.")

    # batched text
    return [w for w in json_data]


@serve.deployment(route_prefix='/')
class SentimentClassifier:
    def __init__(self):
        self.model = model
        self.tokenizer = tokenizer

    def handle(self, inputs: List[str]) -> List[Dict[str, str]]:
        encoded_input = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**encoded_input).logits
        scores = output.detach().numpy()
        probabilities = softmax(scores, axis=1)
        results = []
        for probs in probabilities:
            if entropy(probs) > entropy_threshold:
                results.append({'sentiment': NEUTRAL_SENTIMENT})
                continue
            results.append({'sentiment': mapper[np.argmax(probs)]})
        return results

    async def __call__(self, request: Request):
        js = await request.json()
        return self.handle(get_texts(js))


app = SentimentClassifier.bind()
