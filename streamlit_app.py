import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from scipy.special import softmax
import matplotlib.pyplot as plt
import numpy as np

# Load the sentiment analysis model
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model2 = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)
labels = ['Negative 😞', 'Neutral 😐', 'Positive 😀']

st.title("A STUDY ON SENTIMENTAL ANALYSIS OF MENTAL ILLNESS, CONNOTATIONS OF TEXTS")

# Input area
st.header("Input your tweets here (comma-separated):")
user_input = st.text_area("Enter tweets:")

# Analyze sentiment
if st.button("Predict Sentiment"):
    tweets = user_input.split(',')

    # Lists to store sentiment scores
    negative_scores = []
    neutral_scores = []
    positive_scores = []

    for i, tweet in enumerate(tweets, start=1):
        # Preprocess tweet
        tweet = tweet.strip()  # Remove leading/trailing spaces
        tweet_words = []

        for word in tweet.split(' '):
            if word.startswith('@') and len(word) > 1:
                word = '@user'
            elif word.startswith('http'):
                word = "http"
            tweet_words.append(word)

        tweet_proc = " ".join(tweet_words)

        # Sentiment analysis
        encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
        output = model2(**encoded_tweet)

        scores = output.logits[0].detach().numpy()
        scores = softmax(scores)

        negative_scores.append(scores[0])
        neutral_scores.append(scores[1])
        positive_scores.append(scores[2])

    # Display sentiment scores and create a bar chart
    st.subheader("Sentiment Analysis Results:")
    for i, tweet in enumerate(tweets, start=1):
        st.write(f"**Tweet {i}:** '{tweet}'")
        st.write("Sentiment Scores:")
        st.write(f"- Negative 😞: {negative_scores[i-1]:.5f}")
        st.write(f"- Neutral 😐: {neutral_scores[i-1]:.5f}")
        st.write(f"- Positive 😀: {positive_scores[i-1]:.5f}")

    # Create a bar chart for sentiment scores
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('yellow')
    x = np.arange(len(tweets))
    colors = ['red', 'gray', 'green']
    bar_width = 0.2

    ax.bar(x, negative_scores, width=bar_width, label='Negative 😞', color=colors[0])
    ax.bar(x + bar_width, neutral_scores, width=bar_width, label='Neutral 😐', color=colors[1])
    ax.bar(x + 2 * bar_width, positive_scores, width=bar_width, label='Positive 😀', color=colors[2])

    ax.set_xlabel('Tweets')
    ax.set_ylabel('Sentiment Scores')
    ax.set_title('Sentiment Analysis Results')
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(tweets, rotation=46, ha="right")
    ax.legend()

    # Display the bar chart using st.pyplot()
    st.pyplot(fig)

# Add a clear button to reset the input and results
# Add a clear button to reset the input and results
if st.button("Clear"):
    user_input = ""  # Clear user input
    tweets = []  # Clear stored tweets
    negative_scores = []  # Clear sentiment scores
    neutral_scores = []
    positive_scores = []
    # Clear the text box by updating it with an empty string
    st.text_area("Enter tweets:", value="")

