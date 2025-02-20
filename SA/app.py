from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import praw
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import base64
import io

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Fetch credentials from the .env file
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

# Ensure credentials are loaded
if not client_id or not client_secret:
    raise ValueError("Reddit API credentials are missing in the environment variables.")

user_agent = "script:sentiment-analysis:v1.0 (by /u/Valuable_Bass_7239)"

# Initialize Reddit API
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)

# Load the sentiment analysis pipeline with DistilBERT
model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
sentiment_pipeline = pipeline('sentiment-analysis', model=model_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get the subreddit from the request data
        data = request.get_json()
        subreddit_name = data.get('subreddit')

        if not subreddit_name:
            return jsonify({'error': 'Subreddit name is required'}), 400

        # Fetch top posts from the given subreddit
        subreddit = reddit.subreddit(subreddit_name)
        dfs = []

        # Get the top 10 posts for this year
        top_posts = subreddit.top(time_filter='year', limit=40)

        if not top_posts:
            return jsonify({'error': 'No posts found for this subreddit'}), 400

        # Iterate over the top posts and collect top comments
        for post in top_posts:
            post_title = post.title
            post_url = post.url

            post.comments.replace_more(limit=None)
            top_comments = sorted(post.comments.list(), key=lambda x: x.score, reverse=True)[:5]

            comments_data = {
                'Post_Title': [post_title] * len(top_comments),
                'Post_URL': [post_url] * len(top_comments),
                'Comment_Rank': list(range(1, len(top_comments) + 1)),
                'Comment_Body': [comment.body for comment in top_comments]
            }

            df = pd.DataFrame(comments_data)
            dfs.append(df)

        comments_df = pd.concat(dfs, ignore_index=True)

        # Apply sentiment analysis and handle sentiment correctly
        sentiment_labels = comments_df['Comment_Body'].apply(lambda x: sentiment_pipeline(x)[0]['label'])
        sentiment_counts = sentiment_labels.value_counts().to_dict()

        # Handle missing sentiment counts
        positive_count = sentiment_counts.get('POSITIVE', 0)
        negative_count = sentiment_counts.get('NEGATIVE', 0)

        # Calculate the percentage of positive and negative sentiments
        total_comments = len(comments_df)
        positive_percentage = (positive_count / total_comments) * 100 if total_comments > 0 else 0
        negative_percentage = (negative_count / total_comments) * 100 if total_comments > 0 else 0

        # Create a Matplotlib pie chart
        fig, ax = plt.subplots()
        ax.pie([positive_percentage, negative_percentage], labels=['Positive', 'Negative'], autopct='%1.1f%%',
               colors=['#4CAF50', '#F44336'], startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the chart to a BytesIO object and encode it in base64
        img_io = io.BytesIO()
        plt.savefig(img_io, format='png')
        img_io.seek(0)
        chart_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

        # Return the sentiment counts and the chart image
        return jsonify({
            'positive_count': positive_count,
            'negative_count': negative_count,
            'positive_percentage': positive_percentage,
            'negative_percentage': negative_percentage,
            'chart': chart_base64  # Send the image as a base64 string
        })

    except Exception as e:
        print("Error during analysis:", e)  # Log the full error message
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)