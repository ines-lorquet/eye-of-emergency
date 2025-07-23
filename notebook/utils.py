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
