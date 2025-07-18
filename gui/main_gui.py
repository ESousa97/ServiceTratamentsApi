# gui/main_gui.py

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from core.loader import load_spreadsheet
from core.id_generator import ensure_id_column
from core.utils import validate_file, normalize_cep_column
from analysis.indicator import generate_indicators
from reports.reporter import export_indicators
from config.settings import MAX_ROWS, MAX_FILE_SIZE_MB
import threading
import queue
import textwrap

class SpreadsheetAnalyzerApp:
    def __init__(self, root):
        self.root = root
        root.title("Intelligent Spreadsheet Analyzer")
        root.geometry("1100x800")

        # Seleção de arquivo
        tk.Label(root, text="Selecione a planilha (.csv, .xlsx, .xls):").pack(pady=8)
        frame = tk.Frame(root); frame.pack()
        self.file_entry = tk.Entry(frame, width=95, state='readonly')
        self.file_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(frame, text="Procurar", command=self.select_file).pack(side=tk.LEFT)

        # Botão de analisar
        self.analyze_btn = tk.Button(root, text="Analisar", command=self.start_analysis)
        self.analyze_btn.pack(pady=10)

        # Barra de progresso
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            root, variable=self.progress_var, maximum=100, mode='determinate'
        )
        self.progress_bar.pack(fill='x', padx=10, pady=5)
        self.progress_var.set(0)

        # Área de relatório
        tk.Label(root, text="Relatório:", font=("Arial", 12, "bold")).pack()
        self.output = scrolledtext.ScrolledText(
            root, width=135, height=38, font=('Consolas', 10)
        )
        self.output.pack(padx=8, pady=6)
        self.output.config(state='disabled')

        # Fila de mensagens e estado
        self.msg_queue = queue.Queue()
        self.is_processing = False

        # Agenda a primeira checagem da fila
        self.root.after(100, self.update_gui)

    def select_file(self):
        fn = filedialog.askopenfilename(
            title="Escolha a planilha",
            filetypes=[("Planilhas","*.csv *.xlsx *.xls")]
        )
        if fn:
            self.selected_file = fn
            self.file_entry.config(state='normal')
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, fn)
            self.file_entry.config(state='readonly')

    def start_analysis(self):
        if not getattr(self, 'selected_file', None):
            messagebox.showwarning("Aviso", "Selecione um arquivo primeiro.")
            return

        # Desabilita botão e limpa saída
        self.analyze_btn.config(state='disabled')
        self.output.config(state='normal')
        self.output.delete(1.0, tk.END)
        self.output.config(state='disabled')

        # Zera barra de progresso
        self.progress_var.set(0)
        self.progress_bar.config(mode='determinate')

        # Inicia processamento em thread
        self.is_processing = True
        threading.Thread(target=self.analyze_file, daemon=True).start()

    def update_gui(self):
        # Processa todas as mensagens na fila
        while not self.msg_queue.empty():
            msg, perc = self.msg_queue.get()
            if msg:
                self.output.config(state='normal')
                self.output.insert(tk.END, msg)
                self.output.see(tk.END)
                self.output.config(state='disabled')
            if isinstance(perc, (int, float)):
                self.progress_var.set(perc)

        # Reagenda atualização
        self.root.after(100, self.update_gui)

    def analyze_file(self):
        try:
            validate_file(self.selected_file, MAX_ROWS, MAX_FILE_SIZE_MB)
            ext = self.selected_file.rsplit('.', 1)[1].lower()

            # Carregamento (sem % exato, apenas mensagem)
            self.msg_queue.put(("▸ Carregando planilha...\n", None))
            if ext == 'csv':
                df = load_spreadsheet(self.selected_file, chunksize=50000)
            else:
                df = load_spreadsheet(self.selected_file)

            # Pré‑processamento
            df = ensure_id_column(df)
            df = normalize_cep_column(df)

            # Informa início da análise
            self.msg_queue.put(("▸ Iniciando análise de colunas...\n", 0))

            # Callback de progresso: real-time %
            def analysis_cb(done, total):
                pct = (done / total) * 100
                self.msg_queue.put((f"   Analisando coluna {done}/{total}\n", pct))

            # Gera indicadores
            indicators = generate_indicators(df, progress_callback=analysis_cb)

            # Monta relatório final
            report = []
            report.append("🟦  **Análise da Planilha**\n" + "="*70 + "\n")
            report.append(f"🔹 Coluna de ID:  {indicators['id_coluna']}\n")
            report.append(f"🔹 Linhas:        {indicators['total_linhas']}\n")
            report.append(f"🔹 Colunas:       {indicators['total_colunas']}\n")
            report.append("-"*70 + "\n")

            if not indicators["agrupamentos"]:
                report.append("Nenhuma coluna com valores repetidos para agrupamento.\n")
            else:
                for grp in indicators["agrupamentos"]:
                    report.append(
                        f"🔹 Top agrupamentos para [ {grp['coluna']} ] (tipo: {grp.get('tipo','-')})\n"
                    )
                    if grp.get("tabela") is not None:
                        cols = (
                            ["termo_base","variantes","frequencia"]
                            if "termo_base" in grp["tabela"].columns
                            else ["termo","frequencia"]
                        )
                        sub = grp["tabela"][cols].head(12)
                        report.append(self.format_table(sub, cols) + "\n")
                    elif grp.get("estatisticas"):
                        report.append("• Estatísticas:\n")
                        for k,v in grp["estatisticas"].items():
                            report.append(f"   - {k:<7}: {v}\n")
                    else:
                        report.append("Não há dados agrupados nem estatísticas.\n")
                    report.append("-"*70 + "\n")

            self.msg_queue.put(("".join(report), 100))

            # Exporta CSVs de indicadores
            try:
                export_indicators(indicators, './output')
                self.msg_queue.put(("\n✔ Relatórios salvos em ./output\n", None))
            except Exception as e:
                self.msg_queue.put((f"\n❌ Erro ao salvar relatórios: {e}\n", None))

        except Exception as e:
            self.msg_queue.put((f"\n❌ Erro: {e}\n", None))
        finally:
            # Garante que a barra vá a 100% e reabilita o botão
            self.msg_queue.put((None, 100))
            self.is_processing = False
            self.msg_queue.put((None, None))
            self.analyze_btn.config(state='normal')

    def format_table(self, df, cols, max_var_len=100):
        lines = []
        header = " | ".join([f"{c.upper():<25}" for c in cols])
        lines.append(header)
        lines.append("-" * min(len(header), 150))
        for _, row in df.iterrows():
            vals = []
            for c in cols:
                v = str(row[c]) if c in row else ''
                if c == "variantes" and len(v) > max_var_len:
                    v = textwrap.shorten(v, width=max_var_len, placeholder=" …")
                vals.append(f"{v:<25}")
            lines.append(" | ".join(vals))
        return "\n".join(lines)


def run_gui():
    root = tk.Tk()
    SpreadsheetAnalyzerApp(root)
    root.mainloop()
