import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import GridSearchCV

class ModelTrainer:
    def __init__(self, models):
        self.models = models
        self.trained_models = {}

    def prepare_data(self, path):
        df = pd.read_csv(path, index_col=0)

        def parse_vector(s):
            return np.fromstring(s.strip("[]"), sep=" ")

        text_vectors = np.vstack(df["key_txt_vector"].apply(parse_vector).values)
        numeric_features = df.select_dtypes(include=[np.number]).drop(columns=["target"]).to_numpy()

        scaler = StandardScaler()
        numeric_scaled = scaler.fit_transform(numeric_features)

        X = np.hstack([text_vectors, numeric_scaled])
        y = df["target"].values
        return X, y

    def evaluate_model(self, name, model, X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"\n==== {name} ====")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        # stocker le modèle entraîné
        self.trained_models[name] = model

        # courbe d'apprentissage
        N, train_score, val_score = learning_curve(model, X_train, y_train, cv=4, scoring="f1",
                                                   train_sizes=np.linspace(0.1, 1, 10))
        plt.figure(figsize=(8,5))
        plt.plot(N, train_score.mean(axis=1), label="Train")
        plt.plot(N, val_score.mean(axis=1), label="Validation")
        plt.title(f"Courbe d'apprentissage - {name}")
        plt.xlabel("Taille d'entraînement")
        plt.ylabel("F1 score")
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_trained_model(self, name):
        """Récupère un modèle entraîné par son nom"""
        if name not in self.trained_models:
            raise ValueError(f"Modèle '{name}' non trouvé. Exécute evaluate_model avant.")
        return self.trained_models[name]

    def tune_model(self, name, param_grid, X_train, y_train):
        """Applique un GridSearchCV sur un modèle déjà entraîné"""
        model = self.get_trained_model(name)
        grid = GridSearchCV(model, param_grid, cv=5, scoring="f1", n_jobs=-1)
        grid.fit(X_train, y_train)
        print(f"Meilleurs paramètres pour {name} :", grid.best_params_)
        self.trained_models[name] = grid.best_estimator_
        return grid.best_estimator_

    def save_model(self, name, path=None):
        """Sauvegarde un modèle choisi dans le dossier 'models' à la racine du projet"""
        model = self.get_trained_model(name)
        # Détermine le dossier 'models' à la racine du projet
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(root_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        # Nom du fichier avec timestamp si non fourni
        if path is None:
            timestamp = datetime.now().strftime("%d_%H%M")
            path = os.path.join(models_dir, f"{name}_{timestamp}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"Modèle '{name}' sauvegardé dans {path}")
