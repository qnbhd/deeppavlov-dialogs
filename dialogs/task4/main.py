import pandas as pd
from transformers import pipeline
import gradio as gr

dialogs_df = pd.read_json('../../data/dialogs.json')
subset_dialogs_df = dialogs_df.sample(n=40, random_state=42)


def find_max(scores):
    max_score = 0
    target_label = None

    for row in scores:
        if row['score'] > max_score:
            max_score = row['score']
            target_label = row['label']

    return target_label


# Load the emotion classifier model
classifier = pipeline('text-classification',
                      model='j-hartmann/emotion-english-distilroberta-base',
                      tokenizer='j-hartmann/emotion-english-distilroberta-base')


# Classify emotions in the dialogs
# subset_dialogs_df['emotion'] = subset_dialogs_df['dialogue'].apply(lambda x: find_max(classifier(x)[0]))
subset_dialogs_df['dialogue'] = subset_dialogs_df['dialogue'].map(lambda x: [y for y in x if isinstance(y, str)])
subset_dialogs_df['dialogue'] = subset_dialogs_df['dialogue'].apply(lambda x: ' '.join(x))

# clip for max len
subset_dialogs_df['dialogue'] = subset_dialogs_df['dialogue'].apply(lambda x: x[:512])
subset_dialogs_df['emotion'] = subset_dialogs_df['dialogue'].apply(lambda x: find_max(classifier(x)))

genres = list(set(y for x in dialogs_df['genres'].tolist() for y in x))

print(genres)


# Function to calculate emotion statistics based on genre
def calculate_emotion_statistics(genres):
    # Filter the subset dialogs based on selected genre(s)
    filtered_dialogs_df = subset_dialogs_df[subset_dialogs_df['genres'].map(lambda x: set(x) == set(genres))]

    # Calculate emotion statistics
    emotion_stats = filtered_dialogs_df['emotion'].value_counts().to_dict()

    return emotion_stats


# Interface using Gradio
iface = gr.Interface(
    fn=calculate_emotion_statistics,
    inputs=gr.inputs.CheckboxGroup(genres),
    outputs=gr.outputs.Textbox()
)

# Run the interface
iface.launch()