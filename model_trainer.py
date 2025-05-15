import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load data
df = pd.read_csv("air_quality.csv")

# Feature and target selection
features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
target = 'AQI'

# Drop rows with missing values
df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

# Train model
model = DecisionTreeRegressor()
model.fit(X, y)

# Save model
joblib.dump(model, 'air_quality_model.pkl')
