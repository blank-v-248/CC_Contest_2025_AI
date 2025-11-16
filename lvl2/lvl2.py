import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import math

import os
os.system('cls')
import matplotlib.pyplot as plt
# ---------------------------
# Load the data
# ---------------------------
dfa=pd.read_csv("lvl2/src/level_2_a.in")
dfb=pd.read_csv("lvl2/src/level_2_a.in")
df_ab= pd.concat([dfa, dfb], axis=0, ignore_index=True)
df_ab2 = df_ab[df_ab['Bird Love Score [<3]'] != "missing"]
df0 = pd.read_csv("lvl2/src/level_2_c.in")
df_all= pd.concat([df_ab2, df0], axis=0, ignore_index=True)
df2 = pd.read_csv("lvl2/src/all_data_from_level_1.in")

merged = df_all.merge(df2, on='BOP', how='left')

# Check temperatures for F values
merged['Temperature [°C]'].plot(kind='hist', bins=10, title='Temperature Distribution')
plt.xlabel('Temperature (°C)')
plt.show()

is_fahrenheit = (merged['Temperature [°C]'] > 60)
merged.loc[is_fahrenheit, 'Temperature [°C]'] = (merged.loc[is_fahrenheit, 'Temperature [°C]'] - 32) * 5 / 9
merged['Was_Fahrenheit'] = is_fahrenheit

df = merged
# Replace "missing" with NaN
df["Bird Love Score [<3]"] = pd.to_numeric(df["Bird Love Score [<3]"], errors="coerce")

# ---------------------------
# Split into training (known scores) and prediction (missing)
# ---------------------------
train_df = df[df["Bird Love Score [<3]"].notna()]
predict_df = df[df["Bird Love Score [<3]"].isna()]

X = train_df[["Vegetation [%]", "Insects [g/m²]", "Urban Light [%]", "Temperature [°C]", "Humidity [%]"]]
y = train_df["Bird Love Score [<3]"]

X_pred = predict_df[["Vegetation [%]", "Insects [g/m²]", "Urban Light [%]", "Temperature [°C]", "Humidity [%]"]]
# ---------------------------
# Train/Validation split
# ---------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=2
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_pred_scaled = scaler.transform(X_pred)

# ---------------------------
# Fit regression model FOREST
# ---------------------------
rf = RandomForestRegressor(
    n_estimators=100,   # number of trees
    max_depth=5,     # not too deep to avoid overfitting
    random_state=42
)

rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_train_scaled)
y_pred_val = rf.predict(X_val_scaled)

train_rmse = mean_squared_error(y_train, y_pred)
val_rmse = mean_squared_error(y_val, y_pred_val)
print("Train RMSE:", round(math.sqrt(train_rmse), 3))
print("VAL RMSE:", round(math.sqrt(val_rmse), 3))

# ---------------------------
# Predict missing values
# ---------------------------
predicted = rf.predict(X_pred_scaled)

# Final formatted output
output = pd.DataFrame({
    "BOP": predict_df["BOP"].values,
    "Bird Love Score [<3]": predicted
})

output.to_csv("lvl2/output_c.out", index=False)

