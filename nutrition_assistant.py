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