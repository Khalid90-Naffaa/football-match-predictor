import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

conn = sqlite3.connect("/Users/kh/goglo_Kh_ML/database.sqlite")
df = pd.read_sql("SELECT home_team_goal, away_team_goal, B365H, B365D, B365A, BWH, BWD, BWA FROM Match", conn)
conn.close()

df = df.dropna()

df["result"] = df.apply(
    lambda row: "Home Win" if row["home_team_goal"] > row["away_team_goal"]
    else ("Away Win" if row["home_team_goal"] < row["away_team_goal"] else "Draw"),
    axis=1
)

features = ["B365H", "B365D", "B365A", "BWH", "BWD", "BWA"]
X = df[features]
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2%}")
