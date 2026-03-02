# ==========================================================
# PROJEKTS: Studentu sekmju prognozēšana (stabila versija)
# ==========================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# ==============================
# 1. Datu ielāde
# ==============================

df = pd.read_csv("StudentsPerformance.csv")

print("=== DATU PĀRSKATS ===")
print(df.head())


# ==============================
# 2. Korelācija (tikai skaitliskie dati)
# ==============================

print("\n=== KORELĀCIJA ===")
print(df.select_dtypes(include=['int64']).corr())


# ==============================
# 3. Kategorisko datu pārveidošana skaitļos
# ==============================

# Automātiski pārvērš tekstu par skaitļiem (one-hot encoding)
df = pd.get_dummies(df, drop_first=True)


# ==============================
# 4. Mērķa definēšana
# ==============================

target = "math score"

# Noņemam arī reading un writing,
# lai modelis neizmanto citas atzīmes prognozei
X = df.drop(["math score", "reading score", "writing score"], axis=1)
y = df["math score"]


# ==============================
# 5. Datu sadalīšana
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==============================
# 6. Modeļu salīdzināšana
# ==============================

models = {
    "Lineārā regresija": LinearRegression(),
    "Lēmumu koks": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42)
}

print("\n=== MODEĻU REZULTĀTI ===")

for name, model in models.items():

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"\n{name}")
    print(f"MSE: {mse:.2f}")
    print(f"R2:  {r2:.4f}")


# ==============================
# 7. Faktoru nozīmīgums
# ==============================

best_model = RandomForestRegressor(random_state=42)
best_model.fit(X_train, y_train)

importances = best_model.feature_importances_

importance_df = pd.DataFrame({
    "Faktors": X.columns,
    "Nozīmīgums": importances
}).sort_values(by="Nozīmīgums", ascending=False)

print("\n=== FAKTORU NOZĪMĪGUMS (Random Forest) ===")
print(importance_df)


# ==============================
# 8. Prognozes piemērs
# ==============================

new_student = X.iloc[[0]]
prediction = best_model.predict(new_student)

print("\nPrognozētā Math atzīme:")
print(round(prediction[0], 2))


# ==============================
# 9. Secinājumi
# ==============================

print("\n=== SECINĀJUMI ===")
print("1. Tika analizētas sakarības starp mainīgajiem.")
print("2. Tika salīdzināti trīs mašīnmācīšanās modeļi.")
print("3. Random Forest modelis parasti uzrāda vislabāko precizitāti.")
print("4. Nozīmīgākie faktori redzami augstāk tabulā.")