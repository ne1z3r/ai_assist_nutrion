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
    