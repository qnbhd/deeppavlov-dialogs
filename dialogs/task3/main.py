import numpy as np
import torch
from ray.serve.gradio_integrations import GradioServer
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from scipy.special import softmax
from scipy.stats import entropy
import matplotlib.pyplot as plt
import gradio as gr
import ray

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

np.set_printoptions(suppress=True)

entropy_threshold = 0.693

mapper = ['negative', 'positive']
NEUTRAL_SENTIMENT = 'neutral'


class SentimentClassifier:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def handle(self, inputs):
        encoded_input = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**encoded_input).logits
        scores = output.detach().numpy()
        probabilities = softmax(scores, axis=1)
        results = []
        for probs in probabilities:
            if entropy(probs) > entropy_threshold:
                results.append(NEUTRAL_SENTIMENT)
                continue
            results.append(mapper[np.argmax(probs)])
        return results


@ray.remote
class SentimentAnalyzer:
    def __init__(self):
        self.classifier = SentimentClassifier(
            DistilBertForSequenceClassification.from_pretrained(model_name),
            DistilBertTokenizer.from_pretrained(model_name),
        )

    def analyze_cue(self, cue):
        return self.classifier.handle(cue)

    def analyze_dialog(self, dialog):
        results = []
        for speaker, cues in dialog.items():
            speaker_results = self.analyze_cue(cues)
            results.append((speaker, speaker_results))
        return results


def calculate_sentiment_ratios(dialog):
    speakers = []
    positive_ratios = []
    negative_ratios = []
    neutral_ratios = []
    for speaker, cues in dialog:
        total = len(cues)
        positive_ratio = cues.count('positive') / total if total > 0 else 0
        negative_ratio = cues.count('negative') / total if total > 0 else 0
        neutral_ratio = cues.count('neutral') / total if total > 0 else 0
        speakers.append(speaker)
        positive_ratios.append(positive_ratio)
        negative_ratios.append(negative_ratio)
        neutral_ratios.append(neutral_ratio)
    return speakers, positive_ratios, negative_ratios, neutral_ratios


def convert_text_to_dict(text):
    result_dict = {}
    lines = text.split('\n')
    for line in lines:
        if line.strip() != '':
            key, value = line.split(':')
            key = key.strip()
            value = value.strip()
            if key in result_dict:
                result_dict[key].append(value)
            else:
                result_dict[key] = [value]
    return result_dict


def gradio_sentiment_builder():
    sentiment_analyzer = SentimentAnalyzer.remote()

    def model(text):
        results = ray.get(sentiment_analyzer.analyze_dialog.remote(convert_text_to_dict(text)))
        speakers, positive_ratios, negative_ratios, neutral_ratios = calculate_sentiment_ratios(results)
        ratios = {
            'Positive': positive_ratios,
            'Negative': negative_ratios,
            'Neutral': neutral_ratios
        }

        x = np.arange(len(speakers))
        width = 0.25
        multiplier = 0

        fig, ax = plt.subplots(layout='constrained')

        for attribute, measurement in ratios.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute)
            ax.bar_label(rects, padding=3)
            multiplier += 1

        ax.set_ylabel('Frequency')
        ax.set_title('Sentiment distribution over speakers')
        ax.set_xticks(x + width, speakers)
        ax.legend(loc='upper left', ncols=3)
        ax.set_ylim(0, 1.1)

        return fig

    return gr.Interface(
        fn=model,
        inputs=gr.Textbox(lines=20),
        outputs="plot",
        description="Enter the dialog cues and analyze the sentiment ratios for each speaker.",
        examples=["""a: Hello, my friend
b: Hello, bro"""],
        x='Speakers',
        y=['Positive', 'Negative', 'Neutral'],
        type='bar',
        labels=['Positive', 'Negative', 'Neutral']
    )


app = GradioServer.options().bind(
    gradio_sentiment_builder
)
