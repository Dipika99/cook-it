import streamlit as st
import pandas as pd
import ast
import re
from stable_baselines3 import PPO
from src.env import RecipeEnv
from src.preprocess import load_data

st.set_page_config(
    page_title="Cook It!", 
    layout="centered",
    page_icon="🍳"
)

# Custom CSS for better fonts and styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

.main-header {
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    font-size: 2.5rem;
    color: #1f2937;
    text-align: center;
    margin-bottom: 1rem;
}

.recipe-title {
    font-family: 'Poppins', sans-serif;
    font-weight: 500;
    font-size: 1.3rem;
    color: #374151;
    margin-bottom: 0.5rem;
}

.ingredient-item {
    font-family: 'Poppins', sans-serif;
    font-weight: 400;
    color: #4b5563;
}

.direction-step {
    font-family: 'Poppins', sans-serif;
    font-weight: 400;
    color: #374151;
    line-height: 1.6;
}

.source-text {
    font-family: 'Poppins', sans-serif;
    font-weight: 400;
    color: #6b7280;
    font-style: italic;
}

.stButton > button {
    font-family: 'Poppins', sans-serif;
    font-weight: 500;
    border-radius: 8px;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.recipe-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    border: 1px solid #e5e7eb;
}

.recipe-card:hover {
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    transform: translateY(-2px);
    transition: all 0.3s ease;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🍲 Ready to cook?</h1>', unsafe_allow_html=True)

# === Load data and model ===
@st.cache_data
def get_data():
    return load_data("data/recipes.csv")

@st.cache_resource
def load_model():
    return PPO.load("models/recipe_rl_model")

df = get_data()
model = load_model()

# === State variables ===
if "show_results" not in st.session_state:
    st.session_state.show_results = False
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "clear_input" not in st.session_state:
    st.session_state.clear_input = False

# === Input ===
# Clear input if clear_input flag is set
if st.session_state.clear_input:
    st.session_state.input_text = ""
    st.session_state.clear_input = False

st.text_input(
    "Enter at least 1 ingredient (comma-separated):",
    key="input_text",
    placeholder="e.g. chicken, onion, garlic"
)

col1, col2 = st.columns(2)

with col1:
    if st.button("🍳 Cook It!", use_container_width=True):
        st.session_state.show_results = True

with col2:
    if st.button("🔄 Clear", use_container_width=True):
        st.session_state.clear_input = True
        st.session_state.show_results = False

# === Recipe Logic ===
if st.session_state.show_results and st.session_state.input_text:
    with st.spinner("🍳 Cooking up delicious recipes for you..."):
        ingredients_input = [i.strip().lower() for i in st.session_state.input_text.split(",") if i.strip()]

        if len(ingredients_input) < 1:
            st.warning("Please enter at least 1 ingredient.")
        else:
            matches = []
            for idx in range(len(df)):
                recipe = df.iloc[idx]
                try:
                    recipe_ingredients = set(recipe['NER'])
                except:
                    continue
                match_score = len(set(recipe_ingredients).intersection(set(ingredients_input)))
                if match_score >= 1:
                    try:
                        directions = ast.literal_eval(recipe['directions'])
                        num_steps = len(directions) if isinstance(directions, list) else 999
                    except:
                        num_steps = 999
                    matches.append((idx, match_score, num_steps))

            top_matches = sorted(matches, key=lambda x: (-x[1], x[2]))[:3]

            if top_matches:
                st.markdown('<h2 style="text-align: center; font-family: Poppins;">🎯 Top 3 Recipes You Can Make!</h2>', unsafe_allow_html=True)
                for i, (idx, match_score, num_steps) in enumerate(top_matches, start=1):
                    recipe = df.iloc[idx]
                    st.markdown(f'<div class="recipe-card"><h3 class="recipe-title">{i}. {recipe["title"]}</h3>', unsafe_allow_html=True)

                    st.markdown('<p><b>🥘 Ingredients:</b></p>', unsafe_allow_html=True)
                    try:
                        ingredients_list = ast.literal_eval(recipe['ingredients'])
                        for item in ingredients_list:
                            has = any(ing in item.lower() for ing in ingredients_input)
                            color = "#059669" if has else "#dc2626"
                            icon = "✅" if has else "❌"
                            st.markdown(f'<p class="ingredient-item" style="color:{color};">{icon} {item}</p>', unsafe_allow_html=True)
                    except:
                        st.markdown(recipe['ingredients'])

                    st.markdown('<p><b>📝 Directions:</b></p>', unsafe_allow_html=True)
                    try:
                        directions_data = ast.literal_eval(recipe['directions'])
                    except:
                        directions_data = recipe['directions']

                    if isinstance(directions_data, list):
                        steps = [s.strip() for s in re.split(r'\.\s+', directions_data[0]) if s.strip()]
                        for step_idx, step in enumerate(steps, 1):
                            st.markdown(f'<p class="direction-step">{step_idx}. {step}.</p>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<p class="direction-step">{directions_data}</p>', unsafe_allow_html=True)

                    if recipe.get("link") and str(recipe["link"]).lower() != "nan":
                        st.markdown(f'<p class="source-text">📚 Source: <a href="{recipe["link"]}" target="_blank">{recipe["link"]}</a></p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("No matching recipes found. Try different ingredients.")
