import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from scipy.special import softmax
import matplotlib.pyplot as plt
import numpy as np

def preprocess_tweet(tweet):
    tweet_words = []
    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)
    return " ".join(tweet_words)

def analyze_sentiment(tweet, model, tokenizer):
    tweet_proc = preprocess_tweet(tweet)
    encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
    output = model(**encoded_tweet)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return scores

def main():
    st.title("Twitter Sentiment Analysis")
    st.header("Input your tweets here (comma-separated):")
    user_input = st.text_area("Enter tweets:")

    if st.button("Analyze Sentiment"):
        tweets = user_input.split(',')

        if not tweets[0]:
            st.warning("Please enter at least one tweet.")
            return

        model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        negative_scores, neutral_scores, positive_scores = [], [], []

        for i, tweet in enumerate(tweets, start=1):
            scores = analyze_sentiment(tweet, model, tokenizer)
            negative_scores.append(scores[0])
            neutral_scores.append(scores[1])
            positive_scores.append(scores[2])

        st.subheader("Sentiment Analysis Results:")
        for i, tweet in enumerate(tweets, start=1):
            st.write(f"**Tweet {i}:** '{tweet}'")
            st.write("Sentiment Scores:")
            st.write(f"- Negative 😞: {negative_scores[i-1]:.5f}")
            st.write(f"- Neutral 😐: {neutral_scores[i-1]:.5f}")
            st.write(f"- Positive 😀: {positive_scores[i-1]:.5f}")

        fig, ax = plt.subplots(figsize=(10, 6))
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

        st.pyplot(fig)

if __name__ == "__main__":
    main()
