import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)

df = pd.read_csv('Credit Card Defaulter Prediction.csv')
df.columns = df.columns.str.strip()
df = df.rename(columns={'default': 'DEFAULT'})
df = df.drop(columns=['ID'])

df['EDUCATION'] = df['EDUCATION'].map({
    'Graduate School': 'Graduate School', 'Graduate school': 'Graduate School',
    'University': 'University', 'High School': 'High School'
}).fillna('Others')
df['MARRIAGE'] = df['MARRIAGE'].replace({'0': 'Others'})
df['DEFAULT']  = (df['DEFAULT'] == 'Y').astype(int)

le = LabelEncoder()
for col in ['SEX', 'EDUCATION', 'MARRIAGE']:
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop(columns=['DEFAULT'])
y = df['DEFAULT']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

models = {
    'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), True),
    'Decision Tree':       (DecisionTreeClassifier(max_depth=6, random_state=42), False),
    'Random Forest':       (RandomForestClassifier(n_estimators=200, max_depth=10,
                                                   random_state=42, n_jobs=-1), False),
    'Gradient Boosting':   (GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                        learning_rate=0.1, random_state=42), False),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

SEP  = "═" * 72
SEP_ = "─" * 72

print(f"\n{SEP}")
print(f"  {'MODEL':<24} {'ACC':>7} {'PREC':>7} {'REC':>7} {'F1':>7} {'AUC':>7} {'CV-AUC':>10}")
print(SEP_)

results = {}
for name, (model, scaled) in models.items():
    Xtr, Xte = (X_train_sc, X_test_sc) if scaled else (X_train, X_test)
    model.fit(Xtr, y_train)
    yp    = model.predict(Xte)
    yprob = model.predict_proba(Xte)[:, 1]
    cvauc = cross_val_score(model, Xtr, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

    results[name] = {
        'model':     model,
        'y_pred':    yp,
        'accuracy':  accuracy_score(y_test, yp),
        'precision': precision_score(y_test, yp),
        'recall':    recall_score(y_test, yp),
        'f1':        f1_score(y_test, yp),
        'roc_auc':   roc_auc_score(y_test, yprob),
        'cv_auc':    cvauc.mean(),
        'cv_std':    cvauc.std(),
    }
    r = results[name]
    print(f"  {name:<24} {r['accuracy']:>7.4f} {r['precision']:>7.4f} "
          f"{r['recall']:>7.4f} {r['f1']:>7.4f} {r['roc_auc']:>7.4f} "
          f"{r['cv_auc']:.4f}±{r['cv_std']:.4f}")

print(SEP)

best_name = max(results, key=lambda k: results[k]['roc_auc'])
best      = results[best_name]

print(f"\n  🏆  Best Model : {best_name}")
print(f"      ROC-AUC   : {best['roc_auc']:.4f}")
print(f"      F1-Score  : {best['f1']:.4f}")
print(f"      Accuracy  : {best['accuracy']:.4f}\n")

print(SEP_)
print(f"  Classification Report  —  {best_name}")
print(SEP_)
print(classification_report(y_test, best['y_pred'], target_names=['No Default', 'Default']))

print(SEP_)
print("  Confusion Matrix")
print(SEP_)
cm = confusion_matrix(y_test, best['y_pred'])
print(f"\n               Pred: No Default   Pred: Default")
print(f"  True: No Default      {cm[0,0]:>6,}          {cm[0,1]:>6,}")
print(f"  True: Default         {cm[1,0]:>6,}          {cm[1,1]:>6,}\n")
print(SEP)
