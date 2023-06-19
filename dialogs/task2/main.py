from typing import List, Dict
import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel, validator
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from scipy.special import softmax
from scipy.stats import entropy

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = DistilBertForSequenceClassification.from_pretrained(model_name)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

np.set_printoptions(suppress=True)

entropy_threshold = 0.693

mapper = ['negative', 'positive']
NEUTRAL_SENTIMENT = 'neutral'
MAX_REPLICAS = 10


class InputData(BaseModel):
    texts: List[str]

    @classmethod
    @validator('texts')
    def validate_texts(cls, texts):
        if len(texts) > MAX_REPLICAS:
            raise ValueError("Maximum 10 lines are allowed.")
        return texts


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


app = FastAPI()
classifier = SentimentClassifier()


@app.post("/SentimentClassifier")
def classify_sentiments(data: InputData):
    texts = data.texts
    return classifier.handle(texts)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, port=8001)
