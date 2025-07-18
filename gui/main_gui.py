import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
import subprocess
import requests
import time
import copy

from core.loader import load_spreadsheet
from core.id_generator import ensure_id_column
from core.utils import normalize_cep_column
from analysis.indicator import generate_indicators


def prepare_indicators_for_json(indicators):
    indicators_copy = copy.deepcopy(indicators)
    for grp in indicators_copy.get("agrupamentos", []):
        if grp.get("tabela") is not None:
            grp["tabela"] = grp["tabela"].to_dict(orient="records")
    return indicators_copy


class SpreadsheetAnalyzerApp:
    def __init__(self, root):
        self.root = root
        root.title("Intelligent Spreadsheet Analyzer - Desktop")
        root.geometry("1100x700")

        # Upload file widgets
        tk.Label(root, text="Selecione a planilha (.csv, .xlsx):").pack(pady=6)
        frame = tk.Frame(root)
        frame.pack()
        self.file_entry = tk.Entry(frame, width=90, state='readonly')
        self.file_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(frame, text="Procurar", command=self.select_file).pack(side=tk.LEFT)

        # Botão analisar
        self.analyze_btn = tk.Button(root, text="Analisar", command=self.start_analysis, state='disabled')
        self.analyze_btn.pack(pady=8)

        # Botão abrir interface web (Dash)
        self.web_btn = tk.Button(root, text="Abrir Interface Web", command=self.open_dash_web, state='normal')
        self.web_btn.pack(pady=8)

        # Área resumo
        tk.Label(root, text="Resumo da Análise:", font=("Arial", 12, "bold")).pack()
        self.output = scrolledtext.ScrolledText(root, width=130, height=30, font=('Consolas', 10))
        self.output.pack(padx=10, pady=6)
        self.output.config(state='disabled')

        # Variáveis
        self.selected_file = None
        self.is_processing = False

        # Iniciar dash em background
        threading.Thread(target=self.start_dash_app, daemon=True).start()
        time.sleep(2)  # Aguarda um pouco para o Dash iniciar

    def select_file(self):
        fn = filedialog.askopenfilename(filetypes=[("Planilhas", "*.csv *.xlsx *.xls")])
        if fn:
            self.selected_file = fn
            self.file_entry.config(state='normal')
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, fn)
            self.file_entry.config(state='readonly')
            self.analyze_btn.config(state='normal')

    def start_analysis(self):
        if not self.selected_file or self.is_processing:
            return
        self.is_processing = True
        self.analyze_btn.config(state='disabled')
        self.output.config(state='normal')
        self.output.delete(1.0, tk.END)
        self.output.insert(tk.END, "Iniciando análise...\n")
        self.output.config(state='disabled')
        threading.Thread(target=self.process_file, daemon=True).start()

    def process_file(self):
        try:
            df = load_spreadsheet(self.selected_file)
            df = ensure_id_column(df)
            df = normalize_cep_column(df)
            indicators = generate_indicators(df)

            # Preparar para JSON
            json_ready_indicators = prepare_indicators_for_json(indicators)

            # Construir resumo formatado para Tkinter (texto simples)
            resumo = f"ID Coluna: {indicators['id_coluna']}\nLinhas: {indicators['total_linhas']}\nColunas: {indicators['total_colunas']}\n\n"
            if not indicators["agrupamentos"]:
                resumo += "Nenhuma coluna com valores repetidos para agrupamento.\n"
            else:
                for grp in indicators["agrupamentos"]:
                    resumo += f"🔹 {grp['coluna']} (tipo: {grp.get('tipo', '-')})\n"
                    tabela = grp.get("tabela")
                    if tabela is not None and not tabela.empty:
                        cols = tabela.columns.tolist()
                        termo_col = 'termo_base' if 'termo_base' in cols else ('termo' if 'termo' in cols else None)
                        variantes_col = 'variantes' if 'variantes' in cols else None
                        freq_col = 'frequencia' if 'frequencia' in cols else None

                        header_line = ""
                        if termo_col:
                            header_line += f"{'Termo Base' if termo_col == 'termo_base' else 'Termo':<30} "
                        if variantes_col:
                            header_line += f"{'Variantes':<40} "
                        if freq_col:
                            header_line += f"{'Frequência':<10} "
                        resumo += header_line + "\n"
                        resumo += "-" * len(header_line) + "\n"

                        for _, row in tabela.head(10).iterrows():
                            line = ""
                            if termo_col:
                                termo_val = str(row.get(termo_col, '')).upper()
                                line += f"{termo_val:<30} "
                            if variantes_col:
                                variantes_val = str(row.get(variantes_col, ''))
                                if len(variantes_val) > 38:
                                    variantes_val = variantes_val[:35] + "..."
                                line += f"{variantes_val:<40} "
                            if freq_col:
                                freq_val = str(row.get(freq_col, ''))
                                line += f"{freq_val:<10} "
                            resumo += line + "\n"
                        resumo += "\n"
                    elif grp.get("estatisticas"):
                        resumo += "Estatísticas:\n"
                        for k, v in grp["estatisticas"].items():
                            resumo += f" - {k.capitalize()}: {v}\n"
                        resumo += "\n"
                    else:
                        resumo += "Sem dados agrupados nem estatísticas.\n\n"

            self.output.config(state='normal')
            self.output.delete(1.0, tk.END)
            self.output.insert(tk.END, resumo)
            self.output.config(state='disabled')

            self.send_to_dash(json_ready_indicators)

        except Exception as e:
            self.output.config(state='normal')
            self.output.insert(tk.END, f"Erro na análise: {e}\n")
            self.output.config(state='disabled')

        finally:
            self.is_processing = False
            self.analyze_btn.config(state='normal')

    def send_to_dash(self, indicators):
        url = "http://127.0.0.1:8050/update_data"
        try:
            headers = {'Content-Type': 'application/json'}
            requests.post(url, json=indicators, headers=headers, timeout=5)
        except Exception as e:
            print("Falha ao enviar dados para Dash:", e)

    def open_dash_web(self):
        import webbrowser
        webbrowser.open("http://127.0.0.1:8050")

    def start_dash_app(self):
        import sys
        import os
        import subprocess

        app_py_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
        if os.path.exists(app_py_path):
            subprocess.Popen([sys.executable, app_py_path])
        else:
            print("app.py não encontrado para iniciar o Dash.")


def run_gui():
    root = tk.Tk()
    app = SpreadsheetAnalyzerApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
