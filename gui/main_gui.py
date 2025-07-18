import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from core.loader import load_spreadsheet
from core.id_generator import ensure_id_column
from core.utils import validate_file, normalize_cep_column
from analysis.indicator import generate_indicators
from reports.reporter import export_indicators
from config.settings import MAX_ROWS, MAX_FILE_SIZE_MB
import threading
import textwrap
import queue

def format_table(df, colunas, max_width=80, max_var_len=100):
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

        self.analyze_btn = tk.Button(root, text="Analisar", command=self.start_analysis)
        self.analyze_btn.pack(pady=10)

        # Barra de progresso
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(root, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill='x', padx=10, pady=5)
        self.progress_bar['value'] = 0

        tk.Label(root, text="Relatório:", font=("Arial", 12, "bold")).pack()
        self.output = scrolledtext.ScrolledText(root, width=135, height=38, state='normal', font=('Consolas', 10))
        self.output.pack(padx=8, pady=6)
        self.output.config(state='disabled')

        self.msg_queue = queue.Queue()
        self.is_processing = False

    def select_file(self):
        filetypes = [("Planilhas", "*.csv *.xlsx *.xls")]
        filename = filedialog.askopenfilename(title="Escolha a planilha", filetypes=filetypes)
        if filename:
            self.selected_file = filename
            self.file_entry.config(state='normal')
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, filename)
            self.file_entry.config(state='readonly')

    def update_gui(self):
        updated = False
        while not self.msg_queue.empty():
            msg, perc = self.msg_queue.get()
            self.output.config(state='normal')
            self.output.insert(tk.END, msg)
            self.output.see(tk.END)
            self.output.config(state='disabled')
            if perc is not None:
                self.progress_var.set(perc)
            updated = True
        if self.is_processing:
            self.root.after(100, self.update_gui)
        elif updated:
            self.analyze_btn.config(state='normal')
            self.progress_var.set(0)

    def start_analysis(self):
        if not self.selected_file:
            messagebox.showwarning("Aviso", "Selecione um arquivo primeiro.")
            return
        self.analyze_btn.config(state='disabled')
        self.output.config(state='normal')
        self.output.delete(1.0, tk.END)
        self.progress_var.set(0)
        self.is_processing = True
        threading.Thread(target=self.analyze_file, daemon=True).start()
        self.root.after(100, self.update_gui)

    def analyze_file(self):
        try:
            validate_file(self.selected_file, MAX_ROWS, MAX_FILE_SIZE_MB)
            ext = self.selected_file.split('.')[-1].lower()
            last_perc = -1
            def progress_callback(processed_rows):
                nonlocal last_perc
                perc = min(processed_rows / MAX_ROWS * 100, 100)
                if int(perc) != int(last_perc):
                    dots = '.' * ((int(perc) % 3) + 1)
                    self.msg_queue.put((f"Processando{dots} ({int(perc)}%)\n", perc))
                    last_perc = perc

            if ext == 'csv':
                df = load_spreadsheet(self.selected_file, chunksize=50000, progress_callback=progress_callback)
            else:
                df = load_spreadsheet(self.selected_file)
                progress_callback(len(df))

            if len(df) > MAX_ROWS:
                raise ValueError(f"Limite de {MAX_ROWS} linhas excedido.")

            df = ensure_id_column(df)
            df = normalize_cep_column(df)

            indicators = generate_indicators(df)
            report_text = "🟦  **Análise da Planilha**\n" + "="*70 + "\n"
            report_text += f"🔹 Coluna de ID:  {indicators['id_coluna']}\n"
            report_text += f"🔹 Linhas:        {indicators['total_linhas']}\n"
            report_text += f"🔹 Colunas:       {indicators['total_colunas']}\n"
            report_text += "-"*70 + "\n"

            if not indicators["agrupamentos"]:
                report_text += "\nNenhuma coluna com valores repetidos relevante para agrupamento."
            else:
                for grupo in indicators["agrupamentos"]:
                    report_text += "🔹 Top agrupamentos para:  [ " + grupo["coluna"] + f" ]   (tipo: {grupo.get('tipo','-')})\n"
                    if "tabela" in grupo and grupo["tabela"] is not None:
                        tabela = grupo["tabela"]
                        if "termo_base" in tabela.columns and "variantes" in tabela.columns:
                            cols = ["termo_base", "variantes", "frequencia"]
                        elif "termo" in tabela.columns:
                            cols = ["termo", "frequencia"]
                        else:
                            cols = list(tabela.columns[:3])
                        tabela_exibir = tabela[cols].head(12)
                        if tabela_exibir.empty:
                            report_text += "Nenhum termo encontrado.\n"
                        else:
                            report_text += format_table(tabela_exibir, cols) + "\n"
                    elif "estatisticas" in grupo:
                        stats = grupo["estatisticas"]
                        report_text += "• Estatísticas:\n"
                        for k, v in stats.items():
                            report_text += f"   - {k:<7}: {v}\n"
                    else:
                        report_text += "Não há dados agrupados nem estatísticas.\n"
                    report_text += "-"*70 + "\n"

            self.msg_queue.put((report_text, 100))
            try:
                export_indicators(indicators, './output')
                self.msg_queue.put(("\nRelatórios salvos em ./output\n", None))
            except Exception as e:
                self.msg_queue.put((f"\nErro ao salvar relatórios: {str(e)}\n", None))
        except Exception as e:
            self.msg_queue.put((f"\n❌ Erro ao processar: {str(e)}\n", None))
        finally:
            self.is_processing = False

def run_gui():
    root = tk.Tk()
    app = SpreadsheetAnalyzerApp(root)
    root.mainloop()
