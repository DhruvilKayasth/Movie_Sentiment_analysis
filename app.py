import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="CineSense | Movie Sentiment AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(124, 58, 237, 0.16), transparent 30%),
            radial-gradient(circle at 90% 15%, rgba(236, 72, 153, 0.12), transparent 28%),
            #080b14;
        color: #f8fafc;
    }

    [data-testid="stHeader"] {
        background: rgba(8, 11, 20, 0);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1020 0%, #090b13 100%);
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    .hero {
        padding: 48px 10px 30px 10px;
        text-align: center;
    }

    .badge {
        display: inline-block;
        padding: 7px 14px;
        border-radius: 999px;
        background: rgba(124, 58, 237, 0.14);
        border: 1px solid rgba(167, 139, 250, 0.28);
        color: #c4b5fd;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: .4px;
        margin-bottom: 18px;
    }

    .hero h1 {
        font-size: clamp(38px, 6vw, 68px);
        line-height: 1.05;
        margin: 0;
        font-weight: 800;
        letter-spacing: -2.5px;
        background: linear-gradient(90deg, #ffffff, #c4b5fd 45%, #f9a8d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero p {
        max-width: 720px;
        margin: 18px auto 0;
        color: #9ca3af;
        font-size: 17px;
        line-height: 1.7;
    }

    .glass-card {
        background: rgba(17, 24, 39, 0.68);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 25px;
        box-shadow: 0 20px 50px rgba(0,0,0,.20);
        backdrop-filter: blur(16px);
    }

    .metric-card {
        background: linear-gradient(145deg, rgba(124,58,237,.16), rgba(17,24,39,.70));
        border: 1px solid rgba(167,139,250,.18);
        border-radius: 18px;
        padding: 20px;
        min-height: 120px;
    }

    .metric-label {
        color: #9ca3af;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 30px;
        font-weight: 800;
    }

    .sentiment-positive {
        color: #4ade80;
        font-size: 42px;
        font-weight: 800;
        text-align: center;
    }

    .sentiment-negative {
        color: #fb7185;
        font-size: 42px;
        font-weight: 800;
        text-align: center;
    }

    .sentiment-neutral {
        color: #fbbf24;
        font-size: 42px;
        font-weight: 800;
        text-align: center;
    }

    .confidence {
        text-align: center;
        color: #a1a1aa;
        margin-top: 4px;
    }

    .feature-chip {
        display: inline-block;
        padding: 6px 10px;
        margin: 3px;
        border-radius: 999px;
        background: rgba(255,255,255,.06);
        border: 1px solid rgba(255,255,255,.08);
        color: #cbd5e1;
        font-size: 12px;
    }

    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 12px;
        padding: 35px 0 15px;
    }

    div.stButton > button {
        width: 100%;
        border: 0;
        border-radius: 13px;
        padding: 13px 20px;
        font-weight: 700;
        background: linear-gradient(90deg, #7c3aed, #db2777);
        color: white;
        transition: all .2s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 28px rgba(124,58,237,.30);
    }

    textarea {
        border-radius: 15px !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 9px 16px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL
# Exact pipeline from the supplied notebook
# =========================================================
@st.cache_resource(show_spinner=False)
def train_model():
    data = pd.read_csv("movie_reviews.csv")

    data = data.dropna(subset=["review", "sentiment"]).copy()
    data["review"] = data["review"].astype(str)
    data["sentiment"] = data["sentiment"].astype(str)

    X = data["review"]
    y = data["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(stop_words="english")

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=model.classes_
    )

    return (
        data,
        vectorizer,
        model,
        accuracy,
        cm,
        y_test,
        y_pred
    )


# =========================================================
# HELPERS
# =========================================================
def sentiment_class(label):
    value = str(label).lower()

    if "pos" in value:
        return "positive"
    if "neg" in value:
        return "negative"
    return "neutral"


def sentiment_emoji(label):
    category = sentiment_class(label)

    if category == "positive":
        return "😊"
    if category == "negative":
        return "😞"
    return "😐"


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 🎬 CineSense")
    st.caption("AI-powered movie review analysis")

    st.divider()

    page = st.radio(
        "Navigation",
        ["🔮 Analyze Review", "📊 Model Insights", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### 🧠 Model")
    st.markdown("**Algorithm:** Logistic Regression")
    st.markdown("**Features:** TF-IDF")
    st.markdown("**Split:** 80 / 20")
    st.markdown("**Random State:** 42")

    st.divider()
    st.caption("Built with Python + Scikit-learn + Streamlit")


# =========================================================
# LOAD MODEL
# =========================================================
try:
    (
        data,
        vectorizer,
        model,
        accuracy,
        cm,
        y_test,
        y_pred
    ) = train_model()

except FileNotFoundError:
    st.error(
        "⚠️ `movie_reviews.csv` was not found. "
        "Place the CSV file in the same folder as `app.py`."
    )
    st.stop()

except Exception as e:
    st.error(f"Something went wrong while loading the model: {e}")
    st.stop()


# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="hero">
    <div class="badge">✦ MACHINE LEARNING • NLP • SENTIMENT ANALYSIS</div>
    <h1>Understand the feeling<br>behind every review.</h1>
    <p>
        CineSense uses TF-IDF and Logistic Regression to analyze
        movie reviews and predict their sentiment in seconds.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# PAGE 1 — ANALYZE
# =========================================================
if page == "🔮 Analyze Review":

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">TOTAL REVIEWS</div>
                <div class="metric-value">{len(data):,}</div>
            </div>""",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">MODEL ACCURACY</div>
                <div class="metric-value">{accuracy*100:.1f}%</div>
            </div>""",
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">VOCABULARY</div>
                <div class="metric-value">{len(vectorizer.vocabulary_):,}</div>
            </div>""",
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">CLASSES</div>
                <div class="metric-value">{len(model.classes_)}</div>
            </div>""",
            unsafe_allow_html=True
        )

    st.write("")

    left, right = st.columns([1.25, .75], gap="large")

    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        st.markdown("### ✍️ Analyze a movie review")
        st.caption("Write a review and let the model determine its sentiment.")

        review = st.text_area(
            "Movie review",
            placeholder="Example: The story was amazing, the acting was brilliant, and I loved every minute of it...",
            height=180,
            label_visibility="collapsed"
        )

        examples = [
            "The movie was absolutely fantastic and I loved the story.",
            "This was boring and a complete waste of my time.",
            "The movie was okay, nothing particularly special."
        ]

        selected_example = st.selectbox(
            "Or try an example",
            ["Choose an example..."] + examples
        )

        if selected_example != "Choose an example..." and not review:
            review = selected_example

        analyze = st.button("✨ Analyze Sentiment")

        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        st.markdown("### 🎯 Prediction")

        if analyze:
            if not review or not review.strip():
                st.warning("Please enter a movie review first.")
            else:
                review_tfidf = vectorizer.transform([review])
                prediction = model.predict(review_tfidf)[0]

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(review_tfidf)[0]
                    confidence = float(np.max(probabilities))
                else:
                    confidence = 0.0

                category = sentiment_class(prediction)
                emoji = sentiment_emoji(prediction)

                st.markdown(
                    f'<div class="sentiment-{category}">'
                    f'{emoji} {str(prediction).upper()}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="confidence">'
                    f'Confidence: <b>{confidence*100:.1f}%</b>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.progress(confidence)

                st.write("")

                st.markdown("**Review analyzed:**")
                st.info(review)

                if hasattr(model, "predict_proba"):
                    st.markdown("**Class probabilities**")

                    probability_df = pd.DataFrame({
                        "Sentiment": model.classes_,
                        "Probability": probabilities
                    }).sort_values("Probability", ascending=False)

                    st.bar_chart(
                        probability_df.set_index("Sentiment")
                    )

        else:
            st.markdown(
                """
                <div style="text-align:center; padding:55px 10px;">
                    <div style="font-size:60px;">🍿</div>
                    <h3 style="margin-bottom:8px;">Ready when you are</h3>
                    <p style="color:#9ca3af;">
                        Enter a review on the left and click
                        <b>Analyze Sentiment</b>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# PAGE 2 — MODEL INSIGHTS
# =========================================================
elif page == "📊 Model Insights":

    st.markdown("## 📊 Model Insights")
    st.caption("A quick look at the dataset and the trained classification model.")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Dataset Size", f"{len(data):,}")

    with c2:
        st.metric("Test Samples", f"{len(y_test):,}")

    with c3:
        st.metric("Accuracy", f"{accuracy*100:.2f}%")

    st.write("")

    tab1, tab2, tab3 = st.tabs(
        ["Sentiment Distribution", "Confusion Matrix", "Sample Reviews"]
    )

    with tab1:
        counts = data["sentiment"].value_counts()

        fig, ax = plt.subplots(figsize=(8, 4))
        counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Number of Reviews")
        ax.set_title("Sentiment Distribution")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(cm, interpolation="nearest")
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        ax.set_xticks(range(len(model.classes_)))
        ax.set_yticks(range(len(model.classes_)))
        ax.set_xticklabels(model.classes_)
        ax.set_yticklabels(model.classes_)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")

        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        st.dataframe(
            data[["review", "sentiment"]].head(20),
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# PAGE 3 — ABOUT
# =========================================================
else:
    st.markdown("## ℹ️ About CineSense")

    st.markdown("""
    <div class="glass-card">
        <h3>🎬 What is CineSense?</h3>
        <p style="color:#cbd5e1; line-height:1.8;">
        CineSense is a machine learning application that analyzes the
        sentiment expressed in movie reviews. The original notebook
        uses Natural Language Processing to convert text into numerical
        features and then classifies the review using Logistic Regression.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    cols = st.columns(3)

    with cols[0]:
        st.markdown("""
        <div class="glass-card">
            <h3>01 · TF-IDF</h3>
            <p style="color:#9ca3af;">
            Converts review text into numerical feature vectors based
            on the importance of words.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with cols[1]:
        st.markdown("""
        <div class="glass-card">
            <h3>02 · Logistic Regression</h3>
            <p style="color:#9ca3af;">
            Learns patterns from the training reviews and predicts
            the sentiment of new reviews.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with cols[2]:
        st.markdown("""
        <div class="glass-card">
            <h3>03 · Prediction</h3>
            <p style="color:#9ca3af;">
            A new review is transformed using the same TF-IDF vectorizer
            and passed to the trained model.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown("### 🛠️ Technology Stack")

    technologies = [
        "Python",
        "Pandas",
        "NumPy",
        "Scikit-learn",
        "TF-IDF",
        "Logistic Regression",
        "Streamlit"
    ]

    st.markdown(
        "".join(
            f'<span class="feature-chip">{tech}</span>'
            for tech in technologies
        ),
        unsafe_allow_html=True
    )


# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">
    CineSense · Movie Sentiment Analysis · Built with Machine Learning
</div>
""", unsafe_allow_html=True)
