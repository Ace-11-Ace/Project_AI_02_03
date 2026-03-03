# ==========================================================
# PROJEKTS: Studentu sekmju prognozēšana 
# ==========================================================

# ==============================
# Bibliotēku importēšana
# ==============================

# pandas – bibliotēka darbam ar datu tabulām (DataFrame struktūra)
# Izmanto CSV failu ielādei, datu filtrēšanai un analīzei
import pandas as pd

# numpy – matemātisko aprēķinu bibliotēka
# Nodrošina ātru darbu ar skaitliskajiem masīviem
import numpy as np

# scikit-learn – mašīnmācīšanās bibliotēka
# Satur gatavus algoritmus un rīkus modeļu veidošanai

# Funkcija datu sadalīšanai treniņa un testa kopā
from sklearn.model_selection import train_test_split

# Lineārās regresijas modelis (vienkāršākais regresijas algoritms)
from sklearn.linear_model import LinearRegression

# Lēmumu koka regresijas modelis
from sklearn.tree import DecisionTreeRegressor

# Random Forest regresijas modelis (vairāku lēmumu koku kombinācija)
from sklearn.ensemble import RandomForestRegressor

# Modeļa novērtēšanas metrikas
# mean_squared_error – aprēķina vidējo kvadrātisko kļūdu
# r2_score – aprēķina determinācijas koeficientu (precizitāti)
from sklearn.metrics import mean_squared_error, r2_score


# ==============================
# 1. Datu ielāde
# ==============================

# Ielādējam datu kopu no CSV faila
df = pd.read_csv("StudentsPerformance.csv")

print("=== DATU PĀRSKATS ===")

# Parāda pirmās 5 rindas, lai redzētu datu struktūru
print(df.head())


# ==============================
# 2. Korelācija (tikai skaitliskie dati)
# ==============================

print("\n=== KORELĀCIJA ===")

# Atlasām tikai skaitliskās kolonnas (int64 tips)
# .corr() aprēķina korelācijas matricu starp šīm kolonnām
# Korelācijas vērtība ir no -1 līdz 1
print(df.select_dtypes(include=['int64']).corr())


# ==============================
# 3. Kategorisko datu pārveidošana skaitļos
# ==============================

# Mašīnmācīšanās modeļi nevar strādāt ar teksta vērtībām
# get_dummies() pārveido kategoriskos mainīgos skaitļos (One-Hot Encoding)
# drop_first=True novērš lieku kolonnu dublēšanos
df = pd.get_dummies(df, drop_first=True)


# ==============================
# 4. Mērķa definēšana
# ==============================

# Definējam, kuru mainīgo prognozēsim
target = "math score"

# X – neatkarīgie mainīgie (iezīmes)
# Noņemam reading un writing score,
# lai modelis neizmanto citas atzīmes kā prognozes pamatu
X = df.drop(["math score", "reading score", "writing score"], axis=1)

# y – atkarīgais mainīgais (mērķis)
y = df["math score"]


# ==============================
# 5. Datu sadalīšana
# ==============================

# Sadalam datus 80% treniņam un 20% testam
# random_state=42 nodrošina, ka sadalījums vienmēr būs vienāds
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==============================
# 6. Modeļu salīdzināšana
# ==============================

# Izveidojam trīs dažādus regresijas modeļus salīdzināšanai
models = {
    "Lineārā regresija": LinearRegression(),  # lineāra sakarība starp mainīgajiem
    "Lēmumu koks": DecisionTreeRegressor(random_state=42),  # koka struktūras modelis
    "Random Forest": RandomForestRegressor(random_state=42)  # vairāku koku ansamblis
}

print("\n=== MODEĻU REZULTĀTI ===")

# Cikls, kas apmāca un novērtē katru modeli
for name, model in models.items():

    # Modeļa apmācība uz treniņa datiem
    model.fit(X_train, y_train)

    # Prognozes veikšana uz testa datiem
    predictions = model.predict(X_test)

    # Vidējā kvadrātiskā kļūda (mazāks skaitlis = labāk)
    mse = mean_squared_error(y_test, predictions)

    # R2 koeficients (tuvāk 1 = labāka precizitāte)
    r2 = r2_score(y_test, predictions)

    print(f"\n{name}")
    print(f"MSE: {mse:.2f}")
    print(f"R2:  {r2:.4f}")


# ==============================
# 7. Faktoru nozīmīgums
# ==============================

# Izveidojam jaunu Random Forest modeli
best_model = RandomForestRegressor(random_state=42)

# Apmācām to uz treniņa datiem
best_model.fit(X_train, y_train)

# Iegūstam katras iezīmes ietekmes līmeni
importances = best_model.feature_importances_

# Izveidojam DataFrame ar faktoru nozīmīgumu
importance_df = pd.DataFrame({
    "Faktors": X.columns,
    "Nozīmīgums": importances
}).sort_values(by="Nozīmīgums", ascending=False)

print("\n=== FAKTORU NOZĪMĪGUMS (Random Forest) ===")
print(importance_df)


# ==============================
# 8. Prognozes piemērs
# ==============================

# Izvēlamies pirmo skolēnu no datu kopas
new_student = X.iloc[[0]]

# Prognozējam viņa matemātikas rezultātu
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
