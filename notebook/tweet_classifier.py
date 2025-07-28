import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

class TweetClassifier:
    def __init__(self, model_name="logistic_regression"):
        self.scaler = None
        self.model = self._init_model(model_name)
        
    def _init_model(self, model_name):
        models = {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "decision_tree": DecisionTreeClassifier(),
            "random_forest": RandomForestClassifier(),
            "svm": SVC(),
            "xgboost": XGBClassifier(eval_metric="logloss")
        }
        if model_name not in models:
            raise ValueError(f"Model {model_name} not supported.")
        return models[model_name]

    def _parse_vector(self, s):
        return np.fromstring(s.strip("[]"), sep=" ")

    def prepare_features(self, df):
        if "key_txt_vector" not in df.columns:
            raise ValueError("Key column 'key_txt_vector' is missing from the dataframe")
        text_vectors = np.vstack(df["key_txt_vector"].apply(self._parse_vector).values)

        numeric_features = df.select_dtypes(include=[np.number]).drop(columns=["target"], errors="ignore").to_numpy()
        self.scaler = StandardScaler()
        numeric_scaled = self.scaler.fit_transform(numeric_features)

        X = np.hstack([text_vectors, numeric_scaled])
        y = df["target"].values

        return X, y

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def score(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        return self.model.score(X_test, y_test)
    
    def evaluate_with_learning_curve(self, X_train, y_train, X_test, y_test, scoring="f1"):
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print("=== Évaluation Test ===")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        N, train_score, val_score = learning_curve(
            self.model,
            X_train,
            y_train,
            cv=4,
            scoring=scoring,
            train_sizes=np.linspace(0.1, 1, 10)
        )
        
        plt.figure(figsize=(10, 6))
        plt.plot(N, train_score.mean(axis=1), label="Train score")
        plt.plot(N, val_score.mean(axis=1), label="Validation score")
        plt.title(f"Courbe d'apprentissage ({type(self.model).__name__})")
        plt.xlabel("Taille de l'échantillon d'entraînement")
        plt.ylabel(scoring)
        plt.legend()
        plt.grid(True)
        plt.show()

    def grid_search(self, X_train, y_train, param_grid, scoring="f1", cv=5):
        grid = GridSearchCV(self.model, param_grid, scoring=scoring, cv=cv, n_jobs=-1)
        grid.fit(X_train, y_train)
        self.model = grid.best_estimator_
        return grid.best_params_, grid.best_score_
