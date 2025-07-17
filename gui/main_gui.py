import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from core.loader import load_spreadsheet
from core.id_generator import ensure_id_column
from core.utils import validate_file
from analysis.indicator import generate_indicators
from reports.reporter import export_indicators
from config.settings import MAX_ROWS, MAX_FILE_SIZE_MB
import os
import textwrap

def format_table(df, colunas, max_width=80, max_var_len=100):
    """Formatador visual de tabela (string)"""
    linhas = []
    header = " | ".join([f"{col.upper():<25}" for col in colunas])
    linhas.append(header)
    linhas.append("-" * min(len(header), 150))
    for _, row in df.iterrows():
        values = []
        for col in colunas:
            val = str(row[col]) if col in row else ''
            if col == "variantes" and len(val) > max_var_len:
                val = textwrap.shorten(val, width=max_var_len, placeholder=" ...")
            values.append(f"{val:<25}")
        linhas.append(" | ".join(values))
    return "\n".join(linhas)

class SpreadsheetAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Intelligent Spreadsheet Analyzer")
        self.root.geometry("1100x800")
        self.selected_file = None

        # Widgets
        tk.Label(root, text="Selecione a planilha (.csv, .xlsx, .xls):").pack(pady=8)
        btn_frame = tk.Frame(root)
        btn_frame.pack()
        self.file_entry = tk.Entry(btn_frame, width=95, state='readonly')
        self.file_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Procurar", command=self.select_file).pack(side=tk.LEFT)
        tk.Button(root, text="Analisar", command=self.analyze_file).pack(pady=10)
        tk.Label(root, text="Relatório:", font=("Arial", 12, "bold")).pack()
        self.output = scrolledtext.ScrolledText(root, width=135, height=38, state='normal', font=('Consolas', 10))
        self.output.pack(padx=8, pady=6)
        self.output.config(state='disabled')

    def select_file(self):
        filetypes = [("Planilhas", "*.csv *.xlsx *.xls")]
        filename = filedialog.askopenfilename(title="Escolha a planilha", filetypes=filetypes)
        if filename:
            self.selected_file = filename
            self.file_entry.config(state='normal')
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, filename)
            self.file_entry.config(state='readonly')

    def analyze_file(self):
        self.output.config(state='normal')
        self.output.delete(1.0, tk.END)
        if not self.selected_file:
            messagebox.showwarning("Aviso", "Selecione um arquivo primeiro.")
            return
        try:
            validate_file(self.selected_file, MAX_ROWS, MAX_FILE_SIZE_MB)
            df = load_spreadsheet(self.selected_file)
            if len(df) > MAX_ROWS:
                raise ValueError(f"Limite de {MAX_ROWS} linhas excedido.")
            df = ensure_id_column(df)
            indicators = generate_indicators(df)

            report_text = f"Coluna de ID: {indicators['id_coluna']}\n"
            report_text += f"Linhas: {indicators['total_linhas']}\nColunas: {indicators['total_colunas']}\n"

            if not indicators["agrupamentos"]:
                report_text += "\nNenhuma coluna com valores repetidos relevante para agrupamento."
            else:
                for grupo in indicators["agrupamentos"]:
                    report_text += f"\n{'='*60}\n"
                    report_text += f"Top agrupamentos para '{grupo['coluna']}':\n"
                    tabela = grupo["tabela"]
                    # Decide colunas (universal)
                    if "termo_base" in tabela.columns and "variantes" in tabela.columns:
                        cols = ["termo_base", "variantes", "frequencia"]
                    elif "termo" in tabela.columns:
                        cols = ["termo", "frequencia"]
                    else:
                        cols = list(tabela.columns[:3])
                    tabela_exibir = tabela[cols].head(10)
                    if tabela_exibir.empty:
                        report_text += "Nenhum termo encontrado.\n"
                    else:
                        report_text += format_table(tabela_exibir, cols) + "\n"
            self.output.insert(tk.END, report_text)
            export_indicators(indicators, './output')
            self.output.insert(tk.END, "\nRelatórios salvos em ./output")
        except Exception as e:
            self.output.insert(tk.END, f"Erro ao processar: {str(e)}")
        self.output.config(state='disabled')

def run_gui():
    root = tk.Tk()
    app = SpreadsheetAnalyzerApp(root)
    root.mainloop()
