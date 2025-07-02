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

    def generate_meal(self, meal_type, target_calories):
        """Генерация одного приема пищи"""
        # База рецептов (в реальном приложении должна быть обширной)
        recipes = [
            {"name": "Овсянка с фруктами", "type": "breakfast", "calories": 350},
            {"name": "Тосты с авокадо", "type": "breakfast", "calories": 400},
            {"name": "Куриный салат", "type": "lunch", "calories": 450},
            {"name": "Квиноа с овощами", "type": "lunch", "calories": 400},
            {"name": "Лосось с брокколи", "type": "dinner", "calories": 500},
            {"name": "Овощное рагу", "type": "dinner", "calories": 350},
            {"name": "Греческий йогурт", "type": "snack", "calories": 150},
            {"name": "Ореховая смесь", "type": "snack", "calories": 200}
        ]
        
        # Фильтрация по типу приема пищи
        filtered = [r for r in recipes if r["type"] == meal_type]
        
        # Выбираем рецепт, наиболее близкий к целевым калориям
        closest = min(filtered, key=lambda x: abs(x["calories"] - target_calories), default=None)
        
        return closest or {"name": "Не удалось сгенерировать", "calories": 0}
    
    def plot_nutrition_balance(self):
        """Визуализация баланса питательных веществ"""
        analysis = self.analyze_nutrition_balance()
        if not analysis or not analysis.get("nutrient_comparison"):
            print("Недостаточно данных для визуализации")
            return
        
        comp = analysis["nutrient_comparison"]
        nutrients = list(comp.keys())
        current = [comp[n]["average"] for n in nutrients]
        recommended = [comp[n]["recommended"] for n in nutrients]
        
        x = np.arange(len(nutrients))
        width = 0.35
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, current, width, label='Фактическое')
        plt.bar(x + width/2, recommended, width, label='Рекомендуемое')
        
        plt.title("Баланс питательных веществ")
        plt.xlabel("Питательные вещества")
        plt.ylabel("Количество (г/день)")
        plt.xticks(x, nutrients)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_calorie_trend(self, days=30):
        """Визуализация тренда калорий"""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        dates = []
        calories = []
        
        for i in range(days):
            current_date = start_date + timedelta(days=i)
            daily = self.calculate_daily_nutrition(current_date)
            if daily:
                dates.append(current_date)
                calories.append(daily["total_calories"])
        
        if not dates:
            print("Нет данных для визуализации")
            return
        
        tdee = self.calculate_tdee()
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, calories, 'o-', label='Фактическое потребление')
        plt.axhline(y=tdee, color='r', linestyle='-', label='Рекомендуемое (TDEE)')
        
        # Цели
        for goal in self.diet_goals:
            if goal["type"] in ["weight_loss", "muscle_gain"]:
                target = tdee - 500 if goal["type"] == "weight_loss" else tdee + 300
                plt.axhline(y=target, color='g', linestyle='--', label=f'Цель: {goal["type"]}')
        
        plt.title("Тренд потребления калорий")
        plt.xlabel("Дата")
        plt.ylabel("Калории")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def detect_eating_patterns(self):
        """Обнаружение паттернов питания с помощью ML"""
        if len(self.food_log) < 100:
            print("Недостаточно данных для анализа паттернов")
            return []
        
        # Создаем DataFrame
        df = pd.DataFrame(self.food_log)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['weekday'] = df['timestamp'].dt.weekday
        df['meal_calories'] = df['calories']
        
        # Группировка по приемам пищи
        features = df.groupby(['weekday', 'hour', 'meal_type']).agg({
            'meal_calories': 'mean',
            'food_name': 'count'
        }).reset_index()
        
        # Кластеризация
        X = features[['hour', 'meal_calories']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        features['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Анализ кластеров
        patterns = []
        for cluster in sorted(features['cluster'].unique()):
            cluster_data = features[features['cluster'] == cluster]
            avg_hour = cluster_data['hour'].mean()
            avg_calories = cluster_data['meal_calories'].mean()
            
            pattern = {
                "type": f"Паттерн {cluster+1}",
                "typical_hour": round(avg_hour),
                "typical_calories": round(avg_calories),
                "common_meal_types": cluster_data['meal_type'].mode().tolist()
            }
            patterns.append(pattern)
        
        return patterns

    def generate_report(self):
        """Генерация отчета о питании"""
        analysis = self.analyze_nutrition_balance()
        goals = self.diet_goals
        recommendations = self.get_recommendations()
        
        report = f"Отчет о питании: {self.user_profile['name']}\n"
        report += "=" * 60 + "\n"
        report += f"Дата отчета: {datetime.now().date().isoformat()}\n"
        report += f"Возраст: {self.user_profile['age']} лет, Вес: {self.user_profile['weight']} кг\n"
        report += f"Дневная норма калорий (TDEE): {analysis.get('tdee', 0):.0f} ккал\n"
        report += f"Среднее потребление: {analysis.get('average_calories', 0):.0f} ккал/день\n"
        
        if goals:
            report += "\nТекущие цели:\n"
            for goal in goals:
                status = "Выполнена" if goal.get("completed") else "В процессе"
                report += f"- {goal['type']}: {goal['target_value']} (до {goal['target_date'][:10]}) [{status}]\n"
        
        report += "\nРекомендации:\n"
        for i, rec in enumerate(recommendations, 1):
            report += f"{i}. {rec}\n"
        
        return report