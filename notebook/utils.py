import pandas as pd
import re
import spacy
from tqdm.notebook import tqdm
from nltk.tokenize import RegexpTokenizer


def _target_char(char: str, new_column: str, df: pd.DataFrame, df_column: str) -> pd.DataFrame:
    if df_column not in df.columns:
        raise ValueError(f"La colonne '{df_column}' n'existe pas dans le DataFrame.")
    
    df_result = df.copy()
    
    if new_column not in df_result.columns:
        df_result[new_column] = None
    
    escaped_char = re.escape(char)
    
    pattern = f'{escaped_char}([\\w-]+)'
    
    for idx in df_result.index:
        text = df_result.loc[idx, df_column]
        
        if pd.isna(text) or text is None:
            df_result.loc[idx, new_column] = None
            continue
        
        text_str = str(text)
        matches = re.findall(pattern, text_str)
        cleaned_text = re.sub(escaped_char, '', text_str)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        df_result.loc[idx, df_column] = cleaned_text

        if matches:
            df_result.loc[idx, new_column] = ', '.join(matches)
        else:
            df_result.loc[idx, new_column] = None
    
    return df_result

def reorganize_target_char(df: pd.DataFrame):
    df = _target_char("#", "hashtags", df, "text")
    df = _target_char("@", "mentions", df, "text")
    return df

def extract_url(df: pd.DataFrame, text_column: str = "text", url_column: str = "urls") -> pd.DataFrame:
    if url_column not in df.columns:
        df[url_column] = None

    url_pattern = r"http[s]?://\S+|www\.\S+"

    for idx in df.index:
        text = str(df.loc[idx, text_column]) if pd.notna(df.loc[idx, text_column]) else ""
        urls = re.findall(url_pattern, text)

        cleaned_text = re.sub(url_pattern, "", text).strip()
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

        if urls:
            existing = df.loc[idx, url_column]
            if pd.isna(existing) or existing is None:
                df.loc[idx, url_column] = ", ".join(urls)
            else:
                df.loc[idx, url_column] = existing + ", " + ", ".join(urls)
        else:
            if pd.isna(df.loc[idx, url_column]):
                df.loc[idx, url_column] = None

        df.loc[idx, text_column] = cleaned_text

    return df


def lowercase(df: pd.DataFrame) -> pd.DataFrame:
    df_result = df.copy()
    string_cols = df_result.select_dtypes(include=["object", "string"]).columns
    for col in string_cols:
        df_result[col] = df_result[col].apply(lambda x: x.lower() if isinstance(x, str) else x)
    return df_result

def cleaning_special_char(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    nlp = spacy.load("en_core_web_sm")
    df_result = df.copy()
    for idx in tqdm(df_result.index, desc="Cleaning text", unit="row"):
        text = df_result.loc[idx, text_column]
        if pd.isna(text) or text is None:
            continue
        doc = nlp(str(text))
        tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
        df_result.loc[idx, text_column] = " ".join(tokens)
    return df_result

def clean_text(df: pd.DataFrame, text_column: str='text') -> pd.DataFrame:
    df[text_column] = df[text_column].apply(lambda x: re.sub(r"[^A-Za-z0-9 ]+", " ", str(x)) if pd.notna(x) else x)
    return df

def tokenize_text(df: pd.DataFrame, text_column: str='text', new_column: str='text_tokenize') -> pd.DataFrame:
    tokenizer = RegexpTokenizer(r'\w+')
    df[new_column] = df[text_column].apply(lambda x: tokenizer.tokenize(str(x)) if pd.notna(x) else x)
    return df