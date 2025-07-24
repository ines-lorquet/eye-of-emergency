import pandas as pd
import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer

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
        self.df[string_cols] = self.df[string_cols].apply(lambda col: col.str.lower())
        return self

    def extract_target_char(self, char: str, new_column: str):
        escaped_char = re.escape(char)
        pattern = f'{escaped_char}([\\w-]+)'

        def process_text(text):
            if pd.isna(text):
                return None, text
            text = str(text)
            matches = re.findall(pattern, text)
            if matches:
                cleaned = re.sub(pattern, '', text)
                cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                return ", ".join(matches), cleaned
            else:
                return None, text

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

    def get_df(self):
        return self.df
