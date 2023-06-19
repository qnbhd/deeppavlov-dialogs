import streamlit as st
import pandas as pd
import json

# Placeholder data for demonstration
data = {
    'Movie': ['Movie A', 'Movie A', 'Movie A', 'Movie B'],
    'Speaker': ['A', 'B', 'C', 'D'],
    'Dialogue': [
        ['Hello!', 'How are you?'],
        ['What is your name?'],
        ['Nice to meet you!'],
        ['What the fuck?']
    ],
}

df = pd.read_json('dialogs.json')
df['emotions'] = [[] for _ in range(df.shape[0])]
df['speakers'] = df['speakers'].apply(lambda x: x[0])
df['dialogue'] = df['dialogue'].apply(lambda x: x[:10])
MAX_ROWS = 2

# Movie selection
selected_movie = st.selectbox("Select a movie", df['movie_title'].unique())

if selected_movie:
    movie_df = df[df['movie_title'] == selected_movie]

    st.subheader(f"Annotate replicas for {selected_movie}")

    for i, row in movie_df.iterrows():
        if i == MAX_ROWS:
            break

        dialogue = row['dialogue']

        st.write(f"Speaker: {row['speakers']}")
        for j in range(len(dialogue)):
            previous_line = dialogue[j-1] if j > 0 else ''
            current_line = dialogue[j]

            st.write(f"Previous Line: {previous_line}")
            st.write(f"Current Line: {current_line}")

            emotions = st.multiselect(f"Select emotions that characterize the current line", ["Happy", "Sad", "Angry", "Neutral"], key=f"emotions_{i}_{j}")
            st.write("Selected emotions:", emotions)

            # Store annotations in the DataFrame
            movie_df.at[i, 'emotions'].append({current_line: emotions})

            st.write("---")

    # Export button
    if st.button("Export Annotations"):
        export_data = movie_df[['dialogue', 'emotions', 'speakers']].to_dict(orient='records')
        json_data = json.dumps(export_data, indent=4)
        st.write(f"Exported JSON:\n{json_data}")
