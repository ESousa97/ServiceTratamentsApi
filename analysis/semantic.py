from analysis.stopwords import clean_text, remove_stopwords
import pandas as pd
from collections import defaultdict

def get_terms_frequency(df, text_columns, custom_stopwords=None):
    """
    Retorna um DataFrame com os termos mais frequentes e
    um dicionário mapeando termo -> [ids únicos]
    """
    term_freq = defaultdict(int)
    term_ids = defaultdict(set)
    for idx, row in df.iterrows():
        row_id = row['id']
        for col in text_columns:
            val = clean_text(row[col])
            words = remove_stopwords(val.split(), custom_stopwords)
            for w in words:
                if w.strip():
                    term_freq[w] += 1
                    term_ids[w].add(row_id)
    # DataFrame ordenado por frequência decrescente
    freq_df = pd.DataFrame(
        [(t, term_freq[t], len(term_ids[t])) for t in term_freq],
        columns=['termo', 'frequencia', 'ids_unicos']
    ).sort_values(by=['frequencia', 'termo'], ascending=[False, True]).reset_index(drop=True)
    return freq_df, term_ids
