import pandas as pd
import os
from stable_baselines3 import PPO
from env import RecipeEnv
from preprocess import load_data

# Load and filter data
df = load_data("../data/recipes.csv")

# Create a copy of the dataset and remove source column (keeping original CSV unchanged)
df_clean = df.copy()
if 'source' in df_clean.columns:
    df_clean = df_clean.drop('source', axis=1)
    print("Source column removed from training dataset (original CSV preserved)")

# Example ingredients to train on (can randomize for multiple runs)
sample_ingredients = ['tomato', 'onion', 'garlic']
env = RecipeEnv(df_clean[['NER']], sample_ingredients)

# Create models directory if it doesn't exist
os.makedirs("../models", exist_ok=True)

# Train agent
model = PPO("MlpPolicy", env, verbose=1)
try:
    model.learn(total_timesteps=1000)  # Reduced timesteps for faster training
    model.save("../models/recipe_rl_model")
    print("Model saved successfully!")
    
except KeyboardInterrupt:
    print("Training interrupted, saving model...")
    model.save("../models/recipe_rl_model")
    print("Model saved!")

except Exception as e:
    print(f"Error during training: {e}")
    # Save model even if there's an error
    model.save("../models/recipe_rl_model")
    print("Model saved despite error!")
