import pickle
import numpy as np
import pandas as pd
import torch
import streamlit as st

from models.embedder import load_model
from utils.recommender import compute_similarity

# Set layout
st.set_page_config(layout="wide")

# Load data
series = pickle.load(open('./data/cleaned_series.pkl', 'rb'))
embeddings = np.load('./data/movie_embeddings.npy')
if isinstance(embeddings, np.ndarray):
    embeddings = torch.tensor(embeddings)

# Load model
model = load_model()

# ---- Inject JS to get screen width ----
screen_width_script = """
<script>
const width = window.innerWidth || document.documentElement.clientWidth || document.body.clientWidth;
window.parent.postMessage({screenWidth: width}, "*");
</script>
"""

st.markdown(screen_width_script, unsafe_allow_html=True)

# ---- Receive message from JS ----
if "screen_width" not in st.session_state:
    st.session_state.screen_width = 1200  # default to wide

st.markdown("""
<script>
    window.addEventListener("message", (event) => {
        const width = event.data?.screenWidth;
        if (width) {
            window.parent.postMessage({setSessionState: {screen_width: width}}, "*");
        }
    });
</script>
""", unsafe_allow_html=True)

# ---- UI input ----
is_small_screen = st.session_state.screen_width < 768  # Tailwind 'md' breakpoint

if is_small_screen:
    # STACKED LAYOUT
    st.title('Korean Series Recommender')
    user_input = st.text_input("💬 What kind of story do you want to see?", placeholder="A heartfelt romance between a soldier and a girl...")
    num_input = st.slider("🎯 Number of Recommendations", 1, 10)

    if st.button('🚀 Recommend'):
        if not user_input.strip():
            st.warning("⚠️ Please enter a description or mood first.")
        else:
            top_indices = compute_similarity(user_input, model, embeddings, num_input)
            st.session_state.top_recommendations = series.iloc[top_indices]

    st.subheader("🎥 Top Recommendations:")
    if 'top_recommendations' in st.session_state:
        for idx, row in st.session_state.top_recommendations.iterrows():
            with st.container():
                st.markdown(f"### **{row['title']}** ({int(row['start_year'])})")
                st.markdown(f"⭐ **Rating:** {row['rating']}")
                genres = ', '.join(row['genres']) if isinstance(row['genres'], list) else row['genres']
                stars = ', '.join(row['stars']) if isinstance(row['stars'], list) else row['stars']
                st.markdown(f"🎭 **Genres:** {genres}")
                st.markdown(f"🌟 **Stars:** {stars}")
                if pd.notna(row['description']):
                    st.markdown(f"\n> {row['description']}")
                else:
                    st.markdown("_No description available._")
                st.markdown("---")
    else:
        st.info("Enter a story description and click **Recommend** to see results.")

else:
    # SIDE-BY-SIDE LAYOUT
    col1, spacer, col2 = st.columns([1,0.1, 2])
    with col1:
        st.header('Korean Series Recommender')
        user_input = st.text_input("💬 What kind of story do you want to see?", placeholder="A heartfelt romance between a soldier and a girl...")
        num_input = st.slider("🎯 Number of Recommendations", 1, 10)

        if st.button('🚀 Recommend'):
            if not user_input.strip():
                st.warning("⚠️ Please enter a description or mood first.")
            else:
                top_indices = compute_similarity(user_input, model, embeddings, num_input)
                st.session_state.top_recommendations = series.iloc[top_indices]

    with col2:
        st.subheader("🎥 Top Recommendations:")
        if 'top_recommendations' in st.session_state:
            for idx, row in st.session_state.top_recommendations.iterrows():
                with st.container():
                    st.markdown(f"###  **{row['title']}** ({int(row['start_year'])})")
                    col_left, col_right = st.columns([1, 2])
                    with col_left:
                        genres = ', '.join(row['genres']) if isinstance(row['genres'], list) else row['genres']
                        st.markdown(f"🎭 **Genres:** {genres}")
                    with col_right:
                        stars = ', '.join(row['stars']) if isinstance(row['stars'], list) else row['stars']
                        st.markdown(f"🌟 **Stars:** {stars}")
                    st.markdown(f"⭐ **Rating:** {row['rating']}")
                    if pd.notna(row['description']):
                        st.markdown(f"\n> {row['description']}")
                    else:
                        st.markdown("_No description available._")
                    st.markdown("---")
        else:
            st.info("Enter a story description and click **Recommend** to see results.")
