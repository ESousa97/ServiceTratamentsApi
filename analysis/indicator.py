# analysis/indicator.py

import pandas as pd
from rapidfuzz import fuzz
from analysis.detector import detect_column_types
import unidecode
import re

def is_id_column(col, df):
    return str(col).lower() in ['id', 'codigo', 'identificacao', 'code'] or df[col].is_unique

def is_numerical(col, df):
    return pd.api.types.is_numeric_dtype(df[col])

def is_categorical(col, df):
    if is_numerical(col, df):
        return df[col].nunique() < min(30, len(df) // 5)
    return False

def is_date_candidate(col):
    keywords = ['data', 'date', 'day', 'dia']
    return any(k in unidecode.unidecode(str(col)).lower() for k in keywords)

def normalize_generic(val):
    s = str(val).lower().strip()
    s = unidecode.unidecode(s)
    s = re.sub(r'[^\w\s-]', '', s)
    return s

def safe_to_datetime(series):
    try:
        sample = series.dropna().astype(str).head(50)
        if sample.str.match(r'\d{4}-\d{2}-\d{2}').all():
            return pd.to_datetime(series, format='%Y-%m-%d', errors='coerce')
        else:
            return pd.to_datetime(series, errors='coerce')
    except:
        return pd.to_datetime(series, errors='coerce')

def fuzzy_cluster_terms(terms, threshold=90, max_terms=500):
    if len(terms) > max_terms:
        return [[term] for term in terms]
    clusters, used = [], set()
    for term in terms:
        if term in used: continue
        cluster = [term]; used.add(term)
        for candidate in terms:
            if candidate in used: continue
            if fuzz.ratio(term, candidate) >= threshold:
                cluster.append(candidate)
                used.add(candidate)
        clusters.append(cluster)
    return clusters

def generate_indicators(df, progress_callback=None):
    """
    Gera indicadores e, a cada coluna processada, chama:
        progress_callback(processed_count, total_to_process)
    para streaming de progresso na GUI.
    """
    col_types = detect_column_types(df)
    id_col = next((c for c in df.columns if is_id_column(c, df)), None)
    if not id_col:
        df['_id'] = [str(i) for i in range(1, len(df) + 1)]
        id_col = '_id'

    indicators = {
        "id_coluna": id_col,
        "total_linhas": len(df),
        "total_colunas": len(df.columns),
        "agrupamentos": []
    }
    skip = {id_col}
    to_process = [c for c in df.columns if c not in skip]
    total = len(to_process)
    processed = 0

    for col in to_process:
        label_tipo = col_types.get(col) or "desconhecido"

        # ——— Datas ———
        if is_date_candidate(col):
            conv = safe_to_datetime(df[col])
            indicadores = {
                "coluna": col,
                "tipo": label_tipo,
                "estatisticas": {
                    "min": str(conv.min()),
                    "max": str(conv.max())
                }
            }
            indicators["agrupamentos"].append(indicadores)

        # ——— Numérico contínuo ———
        elif is_numerical(col, df) and not is_categorical(col, df):
            indicadores = {
                "coluna": col,
                "tipo": label_tipo,
                "estatisticas": {
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "media": float(df[col].mean())
                }
            }
            indicators["agrupamentos"].append(indicadores)

        # ——— Categórico ———
        else:
            valores = df[[col, id_col]].dropna()
            vc = valores[col].value_counts()
            if len(vc) > 200:
                top = vc.head(100).index
                valores = valores[valores[col].isin(top)]

            mapping = {}
            for _, row in valores.iterrows():
                orig = str(row[col]).strip()
                norm = normalize_generic(orig)
                rec = mapping.setdefault(norm, {"originais": set(), "ids": set()})
                rec["originais"].add(orig)
                rec["ids"].add(str(row[id_col]))

            clusters = fuzzy_cluster_terms(list(mapping), threshold=88, max_terms=500)
            tabela = []
            for cluster in clusters:
                vars_, ids = set(), set()
                for norm in cluster:
                    vars_.update(mapping[norm]["originais"])
                    ids.update(mapping[norm]["ids"])
                tabela.append({
                    "termo_base": max(cluster, key=len).upper(),
                    "variantes": "; ".join(sorted(vars_)),
                    "frequencia": len(ids),
                    "ids": ",".join(sorted(ids))
                })
            df_tab = pd.DataFrame(tabela).sort_values("frequencia", ascending=False)
            indicadores = {
                "coluna": col,
                "tipo": label_tipo,
                "tabela": df_tab if not df_tab.empty else None
            }
            indicators["agrupamentos"].append(indicadores)

        # Sempre garanta as chaves
        grp = indicators["agrupamentos"][-1]
        grp.setdefault("tabela", None)
        grp.setdefault("estatisticas", None)

        # ——— Progresso ———
        processed += 1
        if progress_callback:
            progress_callback(processed, total)

    return indicators
