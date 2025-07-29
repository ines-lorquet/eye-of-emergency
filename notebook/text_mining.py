import pandas as pd
import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from datetime import datetime
import joblib
import os

class TextMining:
    def __init__(self, df: pd.DataFrame, text_column: str = "text", language: str = "english"):
        self.df = df.copy()
        self.text_column = text_column
        self.language = language
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stop_words = set(stopwords.words(language))
        self.stemmer = SnowballStemmer(language)
        self.lemmatizer = WordNetLemmatizer()
        self.token_column = "tokens"

    def lowercase(self):
        string_cols = self.df.select_dtypes(include=["object", "string"]).columns
        self.df[string_cols] = self.df[string_cols].fillna("")
        self.df[string_cols] = self.df[string_cols].apply(lambda col: col.str.lower())
        return self

    def extract_target_char(self, char: str, new_column: str):
        escaped_char = re.escape(char)
        pattern = f'{escaped_char}([\\w-]+)'
        def process_text(text):
            if pd.isna(text):
                return '', text
            text = str(text)
            matches = re.findall(pattern, text)
            if matches:
                cleaned = re.sub(pattern, '', text)
                cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                return ", ".join(matches), cleaned
            else:
                return '', text
        results = self.df[self.text_column].apply(process_text)
        self.df[new_column] = results.apply(lambda x: x[0])
        self.df[self.text_column] = results.apply(lambda x: x[1])
        return self

    def extract_url(self, url_column: str = "urls"):
        url_pattern = r"http[s]?://\S+|www\.\S+"
        self.df[url_column] = self.df[self.text_column].apply(
            lambda x: ", ".join(re.findall(url_pattern, str(x))) if pd.notna(x) else None
        )
        self.df[self.text_column] = self.df[self.text_column].apply(
            lambda x: re.sub(url_pattern, "", str(x)).strip() if pd.notna(x) else x
        )
        return self

    def clean_regex(self):
        self.df[self.text_column] = self.df[self.text_column].apply(
            lambda x: re.sub(r"[^A-Za-z0-9 ]+", " ", str(x)) if pd.notna(x) else x
        )
        return self

    def tokenize(self):
        self.df[self.token_column] = self.df[self.text_column].apply(
            lambda x: self.tokenizer.tokenize(str(x)) if pd.notna(x) else []
        )
        return self

    def remove_stopwords(self):
        if self.token_column not in self.df:
            raise ValueError("Tokenisation requise avant de retirer les stopwords.")
        self.df[self.token_column] = self.df[self.token_column].apply(
            lambda tokens: [word for word in tokens if word.lower() not in self.stop_words]
        )
        return self

    def apply_stemmer(self):
        if self.token_column not in self.df:
            raise ValueError("Tokenisation requise avant stemming.")
        self.df[self.token_column] = self.df[self.token_column].apply(
            lambda tokens: [self.stemmer.stem(word) for word in tokens]
        )
        return self

    def apply_lemmatizer(self):
        if self.token_column not in self.df:
            raise ValueError("Tokenisation requise avant lemmatisation.")
        self.df[self.token_column] = self.df[self.token_column].apply(
            lambda tokens: [self.lemmatizer.lemmatize(word) for word in tokens]
        )
        return self

    def vectorize(self, mode: str = "bow", new_column: str = "vector"):
        if mode not in ["bow", "tfidf"]:
            raise ValueError("Mode invalide. Utiliser 'bow' ou 'tfidf'.")
        corpus = self.df[self.text_column].fillna("")
        if mode == "bow":
            vectorizer = CountVectorizer()
        else:
            vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(corpus)
        self.df[new_column] = [row for row in vectors.toarray()]
        self.vectorizer = vectorizer  
        return self
        # self.df[new_column] = list(vectors.toarray())
        # self.vectorizer = vectorizer  
        # return self

    def get_model(self, name: str):
        if self.vectorizer is None:
            raise ValueError("Vectorizer no initialized. Call .vectorize() before .get_model()")

        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(root_dir, "models")
        os.makedirs(root_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%d_%H_%M")
        filename = f"{name}_{timestamp}.pkl"
        filepath = os.path.join(data_dir, filename)

        joblib.dump(self.vectorizer, filepath)
        print(f"Vector saved in : {filepath}")
        return filepath

    def export_csv(self, name: str = None):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(root_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        if name is None:
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            name = f"export_{timestamp}.csv"
        path = os.path.join(data_dir, name)
        self.df.to_csv(path, index=False)
        return path


    def get_df(self):
        return self.df
