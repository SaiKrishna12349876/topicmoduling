import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from transformers import pipeline
from sklearn.datasets import fetch_20newsgroups # Import Dataset
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

# --- 1. Page Configuration & Dark Mode CSS ---
st.set_page_config(
    page_title="NarrativeNexus | 20 Newsgroups",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Deep Midnight Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: #e0e0e0;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Neon Headers */
    h1, h2, h3 {
        font-weight: 700;
        background: -webkit-linear-gradient(0deg, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 0px 15px rgba(0, 198, 255, 0.3);
    }
    
    /* Glassmorphism Containers */
    div[data-testid="stExpander"], div[data-testid="stContainer"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        backdrop-filter: blur(10px);
        padding: 15px;
        margin-bottom: 15px;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0a1117; 
        border-right: 1px solid #333;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: rgba(0,0,0,0.2);
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #00c6ff;
    }
    label, p { color: #cfcfcf !important; }
    </style>
""", unsafe_allow_html=True)

# --- 2. Caching Resources & Data ---

@st.cache_resource
def load_nlp_resources():
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@st.cache_data(show_spinner=False)
def load_dataset():
    """
    Loads the 20 Newsgroups dataset automatically.
    Categories are selected based on the user's notebook.
    """
    selected_categories = [
        "comp.graphics", "comp.os.ms-windows.misc", 'alt.atheism',
        "rec.autos", "rec.motorcycles", "rec.sport.baseball",
        'soc.religion.christian', 'talk.politics.mideast',
        "sci.med", "sci.space"
    ]
    # Fetch dataset (cached)
    newsgroups = fetch_20newsgroups(
        subset='all',
        categories=selected_categories,
        remove=('headers', 'footers', 'quotes')
    )
    return newsgroups.data, newsgroups.target_names

summarizer = load_nlp_resources()
lemmatizer = WordNetLemmatizer()

# --- 3. Processing Logic (Gensim) ---

def preprocess_for_gensim(text_list):
    """
    Batch preprocessing for LDA
    """
    processed_data = []
    for text in text_list:
        tokens = simple_preprocess(text, deacc=True)
        clean_tokens = [
            lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in STOPWORDS and len(token) > 3
        ]
        processed_data.append(clean_tokens)
    return processed_data

def train_gensim_lda(clean_docs, num_topics=5):
    # Create Dictionary
    id2word = corpora.Dictionary(clean_docs)
    # Filter extremes to speed up processing on large datasets like Newsgroups
    id2word.filter_extremes(no_below=5, no_above=0.5)
    
    # Create Corpus
    corpus = [id2word.doc2bow(text) for text in clean_docs]
    
    # Train Model
    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        random_state=42,
        update_every=1,
        chunksize=100,
        passes=5, # Reduced passes for interactivity speed
        alpha='auto',
        per_word_topics=True
    )
    return lda_model, corpus, id2word

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# --- 4. Application UI ---

# Sidebar Controls
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.markdown("**Dataset:** 20 Newsgroups (Subset)")
    
    # Load Data Immediately
    with st.spinner("Fetching 20 Newsgroups dataset..."):
        raw_docs, categories = load_dataset()
    
    st.success(f"Loaded {len(raw_docs)} documents")
    st.markdown("---")
    
    num_topics = st.slider("Number of Topics", 2, 15, 5)
    st.markdown("---")
    st.caption("Automated Neural Pipeline")

# Main Header
st.title("📰 NarrativeNexus | 20 Newsgroups")
st.markdown(f"### Analysis of {len(categories)} categories: {', '.join(categories)}")
st.markdown("---")

# --- 5. Global Topic Modeling ---
st.subheader("🔍 Global Topic Modeling")

# Run Preprocessing and LDA
with st.spinner("Preprocessing text & Training LDA Model... (This may take a moment)"):
    # Preprocess all docs
    processed_docs = preprocess_for_gensim(raw_docs)
    
    # Train LDA
    lda_model, corpus, id2word = train_gensim_lda(processed_docs, num_topics=num_topics)

    # Visualization
    cols = st.columns(num_topics)
    
    # If too many topics, create rows
    if num_topics > 4:
        cols = st.columns(4) # Just show 4 per row layout if needed, but Streamlit handles wrapping.
        
    # Loop through topics
    for idx, topic in lda_model.print_topics(-1):
        topic_clean = topic.split('+')
        words = []
        weights = []
        for term in topic_clean:
            weight, word = term.split('*')
            words.append(word.strip().replace('"', ''))
            weights.append(float(weight))
        
        # Add to appropriate column (wrapping)
        col_idx = idx % (len(cols)) if len(cols) > 0 else 0
        
        with cols[col_idx]:
            with st.container(border=True):
                st.markdown(f"<h5 style='color:#00d2ff; text-align:center'>Topic {idx+1}</h5>", unsafe_allow_html=True)
                fig = px.bar(x=weights[:5], y=words[:5], orientation='h', 
                             labels={'x':'Weight', 'y':''},
                             color=weights[:5], color_continuous_scale='Tealgrn')
                fig.update_layout(
                    showlegend=False, height=200, margin=dict(l=0,r=0,t=0,b=0),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="white"), xaxis=dict(showgrid=False, showticklabels=False),
                    yaxis=dict(autorange="reversed"), coloraxis_showscale=False
                )
                st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- 6. Individual Document Explorer ---
st.subheader("📄 Document Explorer")

# Slider to select document index instead of tabs (because there are 10k+ docs)
doc_id = st.slider("Select Document ID", 0, len(raw_docs)-1, 0)

if raw_docs[doc_id]:
    current_text = raw_docs[doc_id]
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown(f"#### Content (ID: {doc_id})")
        
        # Summary
        with st.spinner("Generating Abstractive Summary..."):
            try:
                # Heavily truncate for summarization speed on CPU
                input_text = current_text[:1500] if len(current_text) > 1500 else current_text
                if len(input_text.split()) > 50: # Only summarize if long enough
                    summary_res = summarizer(input_text, max_length=100, min_length=30, do_sample=False)
                    st.markdown(f"""
                    <div style="background-color: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; border-left: 5px solid #00d2ff;">
                        <p style="color: white !important; margin: 0;"><b>Summary:</b> {summary_res[0]['summary_text']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Text too short for summarization.")
            except Exception as e:
                st.warning("Summarization skipped (Text complexity or length issue).")

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📜 Read Full Document", expanded=True):
            st.write(current_text)

    with col2:
        st.markdown("#### Analytics")
        
        # Sentiment
        score = get_sentiment(current_text)
        sent_label = "Positive" if score > 0.1 else "Negative" if score < -0.1 else "Neutral"
        color = "#00d2ff" if score > 0.1 else "#ff4b4b" if score < -0.1 else "#ffa500"

        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"<span style='color:white'>{sent_label}</span>", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [-1, 1], 'tickcolor': "white"},
                'bar': {'color': color},
                'bgcolor': "rgba(255,255,255,0.1)",
                'borderwidth': 2, 'bordercolor': "white",
            }
        ))
        fig_gauge.update_layout(height=200, margin=dict(l=30,r=30,t=50,b=30), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Topic Probability for this specific document
        st.markdown("#### Dominant Topic")
        
        # Get topic distribution for this document from LDA model
        doc_bow = id2word.doc2bow(preprocess_for_gensim([current_text])[0])
        doc_topics = lda_model.get_document_topics(doc_bow)
        
        if doc_topics:
            # Sort by probability
            doc_topics.sort(key=lambda x: x[1], reverse=True)
            top_topic_id, top_prob = doc_topics[0]
            
            st.metric("Topic ID", f"{top_topic_id + 1}", f"{top_prob:.2%} Confidence")
            
            # Show keywords for this topic
            keywords = [word for word, prop in lda_model.show_topic(top_topic_id, topn=5)]
            st.write(f"**Keywords:** {', '.join(keywords)}")
