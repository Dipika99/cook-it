import pandas as pd
import dask.dataframe as dd
import ast

def load_data(path: str, min_ner_len=1):
    df = dd.read_csv(path).compute()
    df['NER'] = df['NER'].apply(ast.literal_eval)
    df = df[df['NER'].map(len) >= min_ner_len]
    df = df.reset_index(drop=True)
    return df
