import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("")

features = [
    "lon",
    "lat",
    "sog",
    "cog",
    "drought",
    "distance_to_port",
    "",
    "",
]
