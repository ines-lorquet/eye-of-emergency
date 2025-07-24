import pandas as pd
import re
from nltk.tokenize import RegexpTokenizer

class TextMining:
    def __init__(self, df: pd.DataFrame, text_column: str = "text"):
        self.df = df.copy()
        self.text_column = text_column
        self.tokenizer = RegexpTokenizer(r'\w+')

    def lowercase(self):
        string_cols = self.df.select_dtypes(include=["object", "string"]).columns
        self.df[string_cols] = self.df[string_cols].apply(lambda col: col.str.lower())
        return self

    def extract_target_char(self, char: str, new_column: str):
        escaped_char = re.escape(char)
        pattern = f'{escaped_char}([\\w-]+)'
        # extraction des cibles
        self.df[new_column] = self.df[self.text_column].apply(
            lambda x: ", ".join(re.findall(pattern, str(x))) if pd.notna(x) else None
        )
        # suppression des symboles
        self.df[self.text_column] = self.df[self.text_column].apply(
            lambda x: re.sub(escaped_char, '', str(x)) if pd.notna(x) else x
        )
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
        # garde seulement lettres, chiffres et espaces
        self.df[self.text_column] = self.df[self.text_column].apply(
            lambda x: re.sub(r"[^A-Za-z0-9 ]+", " ", str(x)) if pd.notna(x) else x
        )
        return self

    def tokenize(self, new_column: str = "tokens"):
        self.df[new_column] = self.df[self.text_column].apply(
            lambda x: self.tokenizer.tokenize(str(x)) if pd.notna(x) else x
        )
        return self

    def get_df(self):
        return self.df
