import json
import os
import re
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import requests
from PIL import Image
from io import BytesIO
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Загрузка ресурсов NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


class NutritionAssistant:
    def __init__(self, data_file="nutrition_data.json", api_key=None):
        self.data_file = data_file
        self.user_profile = {}
        self.food_log = []
        self.diet_goals = []
        self.favorites = []
        self.api_key = api_key  # Ключ для Nutritionix API
        self.lemmatizer = WordNetLemmatizer()
        self.load_data()

    def load_data(self):
        """Загрузка данных из файла"""
        if os.path.exists(self.data_file):
            with open(self.data_file) as f:
                data = json.load(f)
                self.user_profile = data.get("user_profile", {})
                self.food_log = data.get("food_log", [])
                self.diet_goals = data.get("diet_goals", [])
                self.favorites = data.get("favorites", [])
    
    def save_data(self):
        """Сохранение данных в файл"""
        data = {
            "user_profile": self.user_profile,
            "food_log": self.food_log,
            "diet_goals": self.diet_goals,
            "favorites": self.favorites
        }
        with open(self.data_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def setup_profile(self, name, age, gender, height, weight, activity_level, dietary_preferences=None):
        """Настройка профиля пользователя"""
        self.user_profile = {
            "name": name,
            "age": age,
            "gender": gender.lower(),
            "height": height,  # см
            "weight": weight,  # кг
            "activity_level": activity_level,  # sedentary, light, moderate, active, very_active
            "dietary_preferences": dietary_preferences or [],  # vegetarian, vegan, gluten-free, etc.
            "join_date": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        self.save_data()
        return self.user_profile