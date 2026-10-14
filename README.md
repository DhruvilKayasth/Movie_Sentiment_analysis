# CineSense — Movie Sentiment Analysis

Streamlit web app based on the original movie sentiment analysis notebook.

## ML Pipeline
- TF-IDF Vectorizer
- Logistic Regression
- 80/20 stratified train-test split
- random_state=42

## Files required

Keep these files in the same project folder:

- `app.py`
- `movie_reviews.csv`
- `requirements.txt`

The CSV must contain:
- `review`
- `sentiment`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown by Streamlit.
