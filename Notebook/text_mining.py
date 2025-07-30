import pandas as pd
import re
import unicodedata
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from datetime import datetime
import os

class TextMining:
    def __init__(self, df: pd.DataFrame, text_column: str = "text", language: str = "english", tokens_column="tokens"):
        self.df = df.copy()
        self.text_column = text_column
        self.language = language
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stop_words = set(stopwords.words(language))
        self.stemmer = SnowballStemmer(language)
        self.lemmatizer = WordNetLemmatizer()
        self.token_column = tokens_column

    def lowercase(self):
        self.df[self.text_column] = self.df[self.text_column].fillna("").str.lower()
        return self

    def remove_accents(self):
        def strip_accents(text):
            text = unicodedata.normalize('NFD', text)
            text = text.encode('ascii', 'ignore').decode("utf-8")
            return str(text)
        self.df[self.text_column] = self.df[self.text_column].apply(strip_accents)
        return self

    def extract_target_char(self, char: str, new_column: str):
        escaped_char = re.escape(char)
        pattern = f'{escaped_char}([\\w-]+)'
        def process_text(text):
            if pd.isna(text):
                return '', text
            text = str(text)
            matches = re.findall(pattern, text)
            cleaned = re.sub(pattern, '', text)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            return ", ".join(matches), cleaned
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

    def clean_regex(self, columns):
        for col in columns:
            self.df[col] = self.df[col].apply(
                lambda x: re.sub(r"[^A-Za-z0-9 ]+", " ", str(x)) if pd.notna(x) else x
            )
        return self

    def tokenize(self):
        self.df[self.token_column] = self.df[self.text_column].apply(
            lambda x: self.tokenizer.tokenize(str(x)) if pd.notna(x) else []
        )
        return self

    def remove_stopwords(self):
        self.df[self.token_column] = self.df[self.token_column].apply(
            lambda tokens: [word for word in tokens if word.lower() not in self.stop_words]
        )
        return self

    def remove_short_tokens(self, min_len=3):
        self.df[self.token_column] = self.df[self.token_column].apply(
            lambda tokens: [word for word in tokens if len(word) >= min_len]
        )
        return self

    def apply_stemmer(self):
        self.df[self.token_column] = self.df[self.token_column].apply(
            lambda tokens: [self.stemmer.stem(word) for word in tokens]
        )
        return self

    def apply_lemmatizer(self):
        self.df[self.token_column] = self.df[self.token_column].apply(
            lambda tokens: [self.lemmatizer.lemmatize(word) for word in tokens]
        )
        return self

    def build_clean_text(self):
        self.df["clean_text"] = self.df[self.token_column].apply(lambda tokens: " ".join(tokens))
        return self

    def export_csv(self, name: str = None):
        data_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(data_dir, exist_ok=True)
        if name is None:
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            name = f"export_{timestamp}.csv"
        path = os.path.join(data_dir, name)
        self.df.to_csv(path, index=False)
        return path

    def get_df(self):
        return self.df
