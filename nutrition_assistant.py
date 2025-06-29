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

    def set_goal(self, goal_type, target_value, target_date=None):
        """Установка диетической цели"""
        goal = {
            "type": goal_type,  # weight_loss, muscle_gain, calorie_deficit, nutrient_target
            "target_value": target_value,
            "start_date": datetime.now().isoformat(),
            "target_date": target_date or (datetime.now() + timedelta(days=30)).isoformat(),
            "completed": False
        }
        
        if goal_type == "nutrient_target":
            goal["nutrient"] = target_value.get("nutrient")
            goal["target_amount"] = target_value.get("amount")
        
        self.diet_goals.append(goal)
        self.save_data()
        return goal

    def log_food(self, food_name, quantity, unit, meal_type, timestamp=None):
        """Логирование съеденной пищи"""
        # Получаем информацию о пище
        food_info = self.get_food_nutrition(food_name)
        if not food_info:
            raise ValueError(f"Информация о {food_name} не найдена")
        
        # Расчет питательных веществ для количества
        multiplier = quantity / food_info.get("serving_size", 1)
        nutrients = {}
        for nutrient, amount in food_info.get("nutrients", {}).items():
            nutrients[nutrient] = amount * multiplier
        
        log_entry = {
            "food_name": food_name,
            "quantity": quantity,
            "unit": unit,
            "meal_type": meal_type,  # breakfast, lunch, dinner, snack
            "timestamp": timestamp or datetime.now().isoformat(),
            "nutrients": nutrients,
            "calories": food_info.get("calories", 0) * multiplier
        }
        
        self.food_log.append(log_entry)
        self.save_data()
        return log_entry

    def get_food_nutrition(self, food_name):
        """Получение информации о пище (локальная база + API)"""
        # Проверка локальной базы
        local_foods = self.load_food_database()
        for food in local_foods:
            if food_name.lower() in food["name"].lower():
                return food
        
        # Если нет в локальной базе, используем API
        if self.api_key:
            return self.fetch_nutrition_api(food_name)
        
        return None

    def load_food_database(self):
        """Загрузка локальной базы продуктов"""
        # В реальном приложении это может быть внешний файл
        return [
            {
                "name": "Яблоко",
                "calories": 52,
                "serving_size": 100,
                "unit": "г",
                "nutrients": {
                    "protein": 0.3,
                    "fat": 0.2,
                    "carbs": 14,
                    "fiber": 2.4,
                    "sugar": 10,
                    "vitamin_c": 4.6
                }
            },
            {
                "name": "Куриная грудка",
                "calories": 165,
                "serving_size": 100,
                "unit": "г",
                "nutrients": {
                    "protein": 31,
                    "fat": 3.6,
                    "carbs": 0,
                    "iron": 1,
                    "vitamin_b6": 0.6
                }
            },
            {
                "name": "Овсянка",
                "calories": 389,
                "serving_size": 100,
                "unit": "г",
                "nutrients": {
                    "protein": 16.9,
                    "fat": 6.9,
                    "carbs": 66,
                    "fiber": 10.6,
                    "calcium": 54,
                    "iron": 4.7
                }
            }
        ]

    def fetch_nutrition_api(self, food_name):
        """Получение информации о пище через API (Nutritionix)"""
        if not self.api_key:
            return None
        
        url = "https://trackapi.nutritionix.com/v2/natural/nutrients"
        headers = {
            "x-app-id": "YOUR_APP_ID",  # Заменить на реальные
            "x-app-key": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {"query": food_name}
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                data = response.json()
                foods = data.get("foods", [])
                if foods:
                    food = foods[0]
                    nutrients = {
                        "protein": food.get("nf_protein", 0),
                        "fat": food.get("nf_total_fat", 0),
                        "carbs": food.get("nf_total_carbohydrate", 0),
                        "fiber": food.get("nf_dietary_fiber", 0),
                        "sugar": food.get("nf_sugars", 0),
                        "calories": food.get("nf_calories", 0)
                    }
                    return {
                        "name": food.get("food_name", food_name),
                        "calories": food.get("nf_calories", 0),
                        "serving_size": food.get("serving_weight_grams", 100),
                        "unit": "г",
                        "nutrients": nutrients
                    }
        except Exception as e:
            print(f"Ошибка API: {e}")
        
        return None
    
    def add_favorite(self, food_name):
        """Добавление пищи в избранное"""
        food_info = self.get_food_nutrition(food_name)
        if food_info:
            self.favorites.append(food_info)
            self.save_data()
            return food_info
        return None

    def calculate_daily_nutrition(self, date=None):
        """Расчет дневного потребления питательных веществ"""
        date = date or datetime.now().date()
        daily_log = [entry for entry in self.food_log 
                    if datetime.fromisoformat(entry["timestamp"]).date() == date]
        
        if not daily_log:
            return None
        
        total_nutrients = {}
        total_calories = 0
        
        for entry in daily_log:
            total_calories += entry["calories"]
            for nutrient, amount in entry["nutrients"].items():
                if nutrient not in total_nutrients:
                    total_nutrients[nutrient] = 0
                total_nutrients[nutrient] += amount
        
        return {
            "date": date.isoformat(),
            "total_calories": total_calories,
            "nutrients": total_nutrients
        }

    def calculate_bmr(self):
        """Расчет базового метаболизма (BMR)"""
        weight = self.user_profile.get("weight", 70)
        height = self.user_profile.get("height", 170)
        age = self.user_profile.get("age", 30)
        gender = self.user_profile.get("gender", "male")
        
        if gender == "male":
            return 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
        else:
            return 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
    
    def calculate_tdee(self):
        """Расчет общего расхода энергии (TDEE)"""
        bmr = self.calculate_bmr()
        activity_level = self.user_profile.get("activity_level", "moderate")
        
        multipliers = {
            "sedentary": 1.2,
            "light": 1.375,
            "moderate": 1.55,
            "active": 1.725,
            "very_active": 1.9
        }
        
        return bmr * multipliers.get(activity_level, 1.55)

    def analyze_nutrition_balance(self, days=7):
        """Анализ баланса питания за период"""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        # Сбор данных за период
        period_log = [entry for entry in self.food_log 
                     if start_date <= datetime.fromisoformat(entry["timestamp"]).date() <= end_date]
        
        if not period_log:
            return {}
        
        # Расчет средних значений
        avg_calories = sum(entry["calories"] for entry in period_log) / days
        avg_nutrients = {}
        
        # Инициализация питательных веществ
        for entry in period_log:
            for nutrient in entry["nutrients"]:
                if nutrient not in avg_nutrients:
                    avg_nutrients[nutrient] = 0
        
        # Суммирование
        for nutrient in avg_nutrients:
            total = sum(entry["nutrients"].get(nutrient, 0) for entry in period_log)
            avg_nutrients[nutrient] = total / days
        
        # Рекомендуемые нормы (упрощенные)
        recommendations = {
            "protein": 56,  # г
            "fat": 70,      # г
            "carbs": 310,   # г
            "fiber": 25,    # г
            "calcium": 1000, # мг
            "iron": 18      # мг
        }
        
        # Сравнение с рекомендациями
        comparison = {}
        for nutrient, avg_amount in avg_nutrients.items():
            rec = recommendations.get(nutrient, 0)
            if rec > 0:
                percentage = min(150, (avg_amount / rec) * 100)  # Ограничиваем 150%
                status = "low" if percentage < 80 else "high" if percentage > 120 else "optimal"
                comparison[nutrient] = {
                    "average": avg_amount,
                    "recommended": rec,
                    "percentage": percentage,
                    "status": status
                }
        
        return {
            "average_calories": avg_calories,
            "tdee": self.calculate_tdee(),
            "calorie_balance": avg_calories - self.calculate_tdee(),
            "nutrient_comparison": comparison
        }

    def get_recommendations(self):
        """Получение персонализированных рекомендаций"""
        analysis = self.analyze_nutrition_balance()
        goals = self.diet_goals
        recommendations = []
        
        # Рекомендации по калориям
        calorie_balance = analysis.get("calorie_balance", 0)
        main_goal = next((g for g in goals if g["type"] in ["weight_loss", "muscle_gain"]), None)
        
        if main_goal:
            if main_goal["type"] == "weight_loss" and calorie_balance > -200:
                recommendations.append("Снизьте дневное потребление калорий на 200-500 для достижения цели по снижению веса")
            elif main_goal["type"] == "muscle_gain" and calorie_balance < 200:
                recommendations.append("Увеличьте дневное потребление калорий на 200-500 для набора мышечной массы")
        
        # Рекомендации по питательным веществам
        for nutrient, data in analysis.get("nutrient_comparison", {}).items():
            if data["status"] == "low":
                recommendations.append(f"Увеличьте потребление {nutrient}. Текущий уровень: {data['average']:.1f}г/день (рекомендуется: {data['recommended']}г)")
            elif data["status"] == "high":
                recommendations.append(f"Снизьте потребление {nutrient}. Текущий уровень: {data['average']:.1f}г/день (рекомендуется: {data['recommended']}г)")
        
        # Общие рекомендации
        if not recommendations:
            recommendations.append("Ваше питание сбалансировано. Продолжайте в том же духе!")
        
        return recommendations

    def generate_meal_plan(self, calories=None, days=7):
        """Генерация плана питания"""
        target_calories = calories or self.calculate_tdee()
        
        # В реальном приложении здесь была бы сложная логика с учетом предпочтений
        # Для демо используем упрощенный подход
        meal_plan = []
        for day in range(days):
            meals = {
                "breakfast": self.generate_meal("breakfast", target_calories * 0.25),
                "lunch": self.generate_meal("lunch", target_calories * 0.35),
                "dinner": self.generate_meal("dinner", target_calories * 0.30),
                "snacks": self.generate_meal("snack", target_calories * 0.10)
            }
            meal_plan.append({
                "day": (datetime.now() + timedelta(days=day)).date().isoformat(),
                "meals": meals,
                "total_calories": sum(meal["calories"] for meal in meals.values())
            })
        
        return meal_plan