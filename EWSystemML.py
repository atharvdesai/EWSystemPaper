import numpy as np
import pandas as pd
from lazypredict.Supervised import LazyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score


# Load your dataset from 'polygence.xlsx' sheet 1
df = pd.read_csv("polygence.csv")

# Drop irrelevant columns
columns_to_drop = ['date_recorded', 'time_recorded', 'tray_num', 'height_cm', 'water_ml']

# Sort the DataFrame by 'Group' if it's not already sorted
df = df.sort_values(by=['tray_num', 'date_recorded', 'time_recorded'])
df['cumulative_height'] = df.groupby('tray_num')['height_cm'].cumsum()
df['cumulative_water'] = df.groupby('tray_num')['water_ml'].cumsum()
print(df)
# Find the most recent observation (max height and min water consumption) within each group
recent_observation = df.sort_values(['date_recorded','time_recorded'],ascending=False).groupby('tray_num').agg({'cumulative_height': 'max', 'cumulative_water': 'max'})

# Calculate the ratio of max height to min water consumption
recent_observation['Height_to_Water_Ratio'] = recent_observation['cumulative_height'] / recent_observation['cumulative_water']

print(recent_observation)

df = df.drop(columns_to_drop, axis=1)

print(df)
# Separate predictors (X) and outcome (y)
X = df.drop(['cumulative_height'], axis=1)
y = df['cumulative_height']

# Handle missing values with imputation
# imputer = SimpleImputer(strategy='mean')
# X = imputer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

gb_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Fit the model to the training data
gb_regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = gb_regressor.predict(X_test)

rmse_scores = np.sqrt(-cross_val_score(gb_regressor, X, y, cv=10, scoring='neg_mean_squared_error'))

# Calculate the mean RMSE across all folds
mean_rmse = np.mean(rmse_scores)
print(f"Mean Root Mean Squared Error (RMSE): {mean_rmse}")


#
#
# # Create LazyRegressor instance
# reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
#
# # Fit the models and make predictions
# models, predictions = reg.fit(X_train, X_test, y_train, y_test)
#
# print(models)
# models.to_csv("output.csv",index=True)