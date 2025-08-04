import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RecipeEnv(gym.Env):
    def __init__(self, recipes, target_ingredients):
        super().__init__()
        self.recipes = recipes
        self.target = set(target_ingredients)
        self.action_space = spaces.Discrete(len(recipes))
        self.observation_space = spaces.MultiBinary(len(target_ingredients))
        self.current_step = 0

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.done = False
        self.current_step = 0
        return np.zeros(len(self.target), dtype=int), {}

    def step(self, action):
        selected_recipe = self.recipes.iloc[action]
        ingredients = set(selected_recipe['NER'])
        match = self.target.issubset(ingredients)
        reward = 1.0 if match else -1.0
        terminated = True
        truncated = False
        obs = np.array([1 if ing in ingredients else 0 for ing in self.target])
        return obs, reward, terminated, truncated, {}
