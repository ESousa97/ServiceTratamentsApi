import os
import json
import re

def slugify(text):
    # Gera nomes de arquivos seguros
    text = str(text)
    text = re.sub(r'[^\w\s-]', '', text, flags=re.UNICODE)
    text = re.sub(r'[-\s]+', '_', text).strip('_').lower()
    return text or "coluna"

def show_indicators(indicators):
    print("\n🟦  ANÁLISE DA PLANILHA")
    print("="*70)
    print(f"🔹 Coluna de ID:  {indicators.get('id_coluna', '-')}")
    print(f"🔹 Linhas:        {indicators.get('total_linhas', '-')}")
    print(f"🔹 Colunas:       {indicators.get('total_colunas', '-')}")
    print("-"*70)

    if not indicators.get("agrupamentos"):
        print("Nenhuma coluna com valores repetidos relevante para agrupamento.")
        return

    for grupo in indicators["agrupamentos"]:
        col = grupo.get("coluna", "sem_nome")
        tipo = grupo.get("tipo", "-")
        print(f"\n🔹 Top agrupamentos para:  [ {col} ]   (tipo: {tipo})")
        if grupo.get("tabela") is not None:
            tabela = grupo["tabela"]
            if "termo_base" in tabela.columns and "variantes" in tabela.columns:
                cols = ["termo_base", "variantes", "frequencia"]
            elif "termo" in tabela.columns:
                cols = ["termo", "frequencia"]
            else:
                cols = list(tabela.columns[:3])
            print(tabela[cols].head(12).to_string(index=False))
        elif grupo.get("estatisticas"):
            stats = grupo["estatisticas"]
            print("• Estatísticas:")
            for k, v in stats.items():
                print(f"   - {k:<8}: {v}")
        else:
            print("Sem dados agrupados nem estatísticas.")

def export_indicators(indicators, output_dir, base_name="relatorio"):
    os.makedirs(output_dir, exist_ok=True)
    # Exporta meta/resumo
    meta = {
        "id_coluna": indicators.get("id_coluna"),
        "total_linhas": indicators.get("total_linhas"),
        "total_colunas": indicators.get("total_colunas"),
    }
    meta_path = os.path.join(output_dir, f"{base_name}_indicadores.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    for grupo in indicators.get("agrupamentos", []):
        col = grupo.get("coluna", "sem_nome")
        tipo = grupo.get("tipo", "-")
        slug = slugify(col)
        # Exporta tabela agrupada se houver
        tabela = grupo.get("tabela")
        if tabela is not None and hasattr(tabela, "to_csv"):
            tab_path = os.path.join(output_dir, f"{base_name}_{slug}.csv")
            tabela.to_csv(tab_path, index=False, encoding='utf-8')
            print(f"[OK] Exportado agrupamento: {tab_path}")
        # Exporta estatísticas se houver
        stats = grupo.get("estatisticas")
        if stats is not None:
            stat_path = os.path.join(output_dir, f"{base_name}_{slug}_estatisticas.txt")
            with open(stat_path, "w", encoding="utf-8") as f:
                for k, v in stats.items():
                    f.write(f"{k}: {v}\n")
            print(f"[OK] Exportado estatísticas: {stat_path}")
