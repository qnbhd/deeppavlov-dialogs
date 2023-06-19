from collections import defaultdict

import streamlit as st
import pandas as pd
from transformers import pipeline

classifier = pipeline('text-classification',
                      model='j-hartmann/emotion-english-distilroberta-base',
                      tokenizer='j-hartmann/emotion-english-distilroberta-base', return_all_scores=True)


df = pd.read_json('../../data/dialogs.json')
df['speakers'] = df['speakers'].apply(lambda x: x[0])


def detect_for_speaker(rows, speaker):
    text = ' '.join(rows[rows['speakers'] == speaker].iloc[0]['dialogue'])
    text = text[:512]
    result = classifier(text)
    print(result)
    w = {}
    for u in result[0]:
        w[u['label']] = [u['score']]
    return w


# Movie selection
selected_movie = st.selectbox("Select a movie", df['movie_title'].unique())


if selected_movie:
    # movie_row = df[df['movie_title'] == selected_movie].iloc[0]
    movie_rows = df[df['movie_title'] == selected_movie]
    movie_rows = movie_rows[movie_rows['dialogue'].map(lambda x: all(isinstance(y, str) for y in x))]
    selected_speaker = st.selectbox("Select a speaker", ["All"] + list(movie_rows['speakers'].unique()))

    if selected_speaker != "All":
        result = detect_for_speaker(movie_rows, selected_speaker)
        st.write(f"Emotion statistics for {selected_speaker} in {selected_movie}:")
        st.write(pd.DataFrame(result))
    else:
        st.write(f"Emotion statistics for all speakers in {selected_movie}:")
        buff = defaultdict(list)
        for speaker in list(movie_rows['speakers'].unique()):
            result = detect_for_speaker(movie_rows, speaker)
            buff['speaker'].append(speaker)
            for k, v in result.items():
                buff[k].append(v)
        st.write(pd.DataFrame(buff))
