import shap
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb

# Charger les données non normalisées
X_train = pd.read_csv("../data/X_train_original.csv")
X_test = pd.read_csv("../data/X_test_original.csv")
y_train = pd.read_csv("../data/y_train.csv").values.ravel()  # Convertir en 1D
y_test = pd.read_csv("../data/y_test.csv").values.ravel()

# Charger et entraîner le meilleur modèle
best_model = lgb.LGBMClassifier()
best_model.fit(X_train, y_train)

# Initialiser SHAP avec l'option check_additivity=False
explainer = shap.Explainer(best_model, X_train)
shap_values = explainer(X_test, check_additivity=False)  # Corrige l'erreur d'additivité

# Afficher l'importance des features
shap.summary_plot(shap_values, X_test, show=True)
