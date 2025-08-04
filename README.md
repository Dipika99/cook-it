# 🍳 Cook It! — Ingredient-Based Recipe Recommender

**Cook It!** is a Streamlit-based web app that recommends simple, tasty recipes based on the ingredients you already have. Using a reinforcement learning model, it prioritizes matches with fewer steps and more overlap with your kitchen items.

---

## Features

- **Ingredient-Based Search**: Enter ingredients you have, and get recipe recommendations.
- **Smart Ranking**: Recipes are ranked based on matching ingredients and simplicity.
- **Clear and Intuitive UI**: Easy input and visually distinct results.
- 
## ⚙️ How to Install & Run

1. **Clone this repo**:

   ```bash
   git clone https://github.com/yourusername/cook-it.git
   cd cook-it
   
2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   
3. **Run the training script**:

   ```bash
   python train_rl.py
   
3. **Run the training script**:

   ```bash
   streamlit run app.py

## Dataset & Credits

This project uses recipe data from:

- [RecipeNLG Dataset](https://www.kaggle.com/datasets/paultimothymooney/recipenlg) by Paul Timothy Mooney  
- [Custom Recipe Dataset](https://www.kaggle.com/code/anglerr/custom-recipe-dataset) by Angler  

These datasets are used for educational and non-commercial purposes. Full credit to the original dataset authors.

