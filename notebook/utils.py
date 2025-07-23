import pandas as pd
import re
def special_char(char: str, new_column: str, df: pd.DataFrame, df_column: str) -> pd.DataFrame:
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

import pandas as pd
import re

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

import pandas as pd

def lowercase(df: pd.DataFrame) -> pd.DataFrame:
    df_result = df.copy()
    string_cols = df_result.select_dtypes(include=["object", "string"]).columns
    for col in string_cols:
        df_result[col] = df_result[col].apply(lambda x: x.lower() if isinstance(x, str) else x)
    return df_result
