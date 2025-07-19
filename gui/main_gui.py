import sys
import os
import json
import threading
import webbrowser
import subprocess
import requests
import qtawesome as qta
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextBrowser, QProgressBar, QFileDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from core.loader import load_spreadsheet
from core.id_generator import ensure_id_column
from core.utils import normalize_cep_column
from analysis.indicator import generate_indicators


def prepare_indicators_for_json(indicators):
    copy = {**indicators}
    agrup = []
    for grp in indicators.get('agrupamentos', []):
        entry = dict(grp)
        if entry.get('tabela') is not None:
            entry['tabela'] = entry['tabela'].to_dict(orient='records')
        agrup.append(entry)
    copy['agrupamentos'] = agrup
    return copy


class AnalyzeWorker(QThread):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath

    def run(self):
        try:
            df = load_spreadsheet(self.filepath, progress_callback=self._on_progress)
            df = ensure_id_column(df)
            df = normalize_cep_column(df)
            indicators = generate_indicators(df, progress_callback=self._on_progress)
            self.finished.emit(indicators)
        except Exception as e:
            self.error.emit(str(e))

    def _on_progress(self, processed, total=None):
        if total:
            self.progress.emit(processed, total)
        else:
            self.progress.emit(processed, processed)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Intelligent Spreadsheet Analyzer - Desktop')
        self.setGeometry(200, 200, 1100, 700)
        self._apply_styles()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Título com ícone
        title_layout = QHBoxLayout()
        icon_label = QLabel()
        icon_label.setPixmap(qta.icon('fa5s.chart-bar', color='#7289DA').pixmap(26, 26))
        title_layout.addWidget(icon_label)
        title = QLabel("Resumo da Análise")
        title.setStyleSheet('font-weight: bold; font-size: 19px; color: #7289DA;')
        title_layout.addWidget(title)
        title_layout.addStretch()
        layout.addLayout(title_layout)

        # File selection
        file_layout = QHBoxLayout()
        excel_icon = QLabel()
        excel_icon.setPixmap(qta.icon('fa5s.file-excel', color='#43B581').pixmap(20, 20))
        file_label = QLabel('Selecione a planilha (.csv, .xlsx, .xls):')
        file_layout.addWidget(excel_icon)
        file_layout.addWidget(file_label)
        self.file_entry = QLineEdit()
        self.file_entry.setReadOnly(True)
        browse_btn = QPushButton(qta.icon('fa5s.folder-open', color='#fff'), ' Procurar')
        browse_btn.clicked.connect(self.select_file)
        file_layout.addWidget(self.file_entry)
        file_layout.addWidget(browse_btn)
        layout.addLayout(file_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        self.analyze_btn = QPushButton(qta.icon('fa5s.play', color='#fff'), ' Analisar')
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.web_btn = QPushButton(qta.icon('fa5s.globe', color='#fff'), ' Abrir Interface Web')
        self.web_btn.clicked.connect(lambda: webbrowser.open('http://127.0.0.1:8050'))
        btn_layout.addWidget(self.analyze_btn)
        btn_layout.addWidget(self.web_btn)
        layout.addLayout(btn_layout)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # Info resumo (ID, linhas, colunas) com ícones
        info_layout = QHBoxLayout()
        self.label_id = QLabel()
        self.label_id.setStyleSheet("font-size: 13px; color: #43B581;")
        self.label_rows = QLabel()
        self.label_rows.setStyleSheet("font-size: 13px; color: #FFD700;")
        self.label_cols = QLabel()
        self.label_cols.setStyleSheet("font-size: 13px; color: #FFD700;")
        id_icon = QLabel()
        id_icon.setPixmap(qta.icon('fa5s.id-badge', color='#43B581').pixmap(18, 18))
        row_icon = QLabel()
        row_icon.setPixmap(qta.icon('fa5s.list-ol', color='#FFD700').pixmap(18, 18))
        col_icon = QLabel()
        col_icon.setPixmap(qta.icon('fa5s.table', color='#FFD700').pixmap(18, 18))
        info_layout.addWidget(id_icon)
        info_layout.addWidget(self.label_id)
        info_layout.addWidget(row_icon)
        info_layout.addWidget(self.label_rows)
        info_layout.addWidget(col_icon)
        info_layout.addWidget(self.label_cols)
        info_layout.addStretch()
        layout.addLayout(info_layout)

        # Summary HTML
        self.output = QTextBrowser()
        self.output.setOpenExternalLinks(True)
        self.output.setReadOnly(True)
        layout.addWidget(self.output)

        # Footer / Rodapé elegante e centralizado com botões LinkedIn e GitHub
        footer = self.create_footer()
        layout.addWidget(footer)

        # Dash background starter
        threading.Thread(target=self._start_dash, daemon=True).start()

    def create_footer(self):
        rodape = QWidget()
        rodape_layout = QVBoxLayout(rodape)
        rodape_layout.setContentsMargins(0, 10, 0, 0)
        rodape_layout.setSpacing(4)

        # Linha separadora
        line = QLabel()
        line.setFixedHeight(1)
        line.setStyleSheet("background: #23272A; margin-bottom: 4px;")
        rodape_layout.addWidget(line)

        # Conteúdo centralizado
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        center_layout.setSpacing(2)

        # Linha 1: Nome e profissão
        nome_layout = QHBoxLayout()
        nome_layout.setAlignment(Qt.AlignHCenter)
        dev_icon = QLabel()
        dev_icon.setPixmap(qta.icon('fa5s.user-tie', color='#7289DA').pixmap(18, 18))
        dev_text = QLabel("<b>José Enoque</b>  —  Desenvolvedor Full Stack")
        dev_text.setStyleSheet("color:#B9BBBE; font-size:13px; margin-left:3px;")
        nome_layout.addWidget(dev_icon)
        nome_layout.addWidget(dev_text)
        center_layout.addLayout(nome_layout)

        # Linha 2: Foco em automação
        auto_layout = QHBoxLayout()
        auto_layout.setAlignment(Qt.AlignHCenter)
        auto_icon = QLabel()
        auto_icon.setPixmap(qta.icon('fa5s.robot', color='#43B581').pixmap(17, 17))
        auto_text = QLabel("Foco em automação e soluções inteligentes")
        auto_text.setStyleSheet("color:#8bffae; font-size:12px; font-style:italic; margin-left:3px;")
        auto_layout.addWidget(auto_icon)
        auto_layout.addWidget(auto_text)
        center_layout.addLayout(auto_layout)

        # Botões de rede social
        btns_layout = QHBoxLayout()
        btns_layout.setAlignment(Qt.AlignHCenter)
        
        linkedin_btn = QPushButton(qta.icon('fa5b.linkedin', color='#0A66C2'), "LinkedIn")
        linkedin_btn.setStyleSheet("""
            QPushButton {
                background-color: #23272A;
                color: #0A66C2;
                border: 1px solid #0A66C2;
                border-radius: 5px;
                padding: 2px 16px 2px 8px;
                font-size: 12px;
                margin: 5px 2px 0 2px;
            }
            QPushButton:hover {
                background-color: #0A66C2;
                color: #fff;
            }
        """)
        linkedin_btn.clicked.connect(lambda: webbrowser.open('https://www.linkedin.com/in/enoque-sousa-bb89aa168/'))
        btns_layout.addWidget(linkedin_btn)

        github_btn = QPushButton(qta.icon('fa5b.github', color='#b9bbbe'), "GitHub")
        github_btn.setStyleSheet("""
            QPushButton {
                background-color: #23272A;
                color: #b9bbbe;
                border: 1px solid #b9bbbe;
                border-radius: 5px;
                padding: 2px 16px 2px 8px;
                font-size: 12px;
                margin: 5px 2px 0 2px;
            }
            QPushButton:hover {
                background-color: #b9bbbe;
                color: #23272A;
            }
        """)
        github_btn.clicked.connect(lambda: webbrowser.open('https://github.com/ESousa97'))
        btns_layout.addWidget(github_btn)

        center_layout.addLayout(btns_layout)

        rodape_layout.addWidget(center_widget)
        return rodape

    def _apply_styles(self):
        self.setStyleSheet("""
        QMainWindow {
            background-color: #36393F;
        }
        QLabel {
            color: #FFFFFF;
        }
        QPushButton {
            background-color: #5865F2;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-size: 14px;
        }
        QPushButton:hover {
            background-color: #4752C4;
        }
        QPushButton:disabled {
            background-color: #4E5058;
            color: #B9BBBE;
        }
        QLineEdit {
            background-color: #2F3136;
            color: #FFFFFF;
            border: 1px solid #202225;
            border-radius: 4px;
            padding: 4px;
        }
        QTextBrowser {
            background-color: #2F3136;
            color: #FFFFFF;
            border: 1px solid #202225;
            border-radius: 4px;
            font-family: Consolas;
            font-size: 12px;
        }
        QProgressBar {
            background-color: #202225;
            color: #FFFFFF;
            border: 1px solid #202225;
            border-radius: 4px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #7289DA;
            border-radius: 4px;
        }
        """)

    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Selecionar Planilha', os.getcwd(), 'Planilhas (*.csv *.xlsx *.xls)'
        )
        if path:
            self.file_entry.setText(path)
            self.analyze_btn.setEnabled(True)
            self.filepath = path

    def start_analysis(self):
        self.analyze_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.output.clear()
        self.worker = AnalyzeWorker(self.filepath)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.error.connect(self.show_error)
        self.worker.start()

    def update_progress(self, processed, total):
        self.progress.setMaximum(total)
        self.progress.setValue(processed)

    def analysis_finished(self, indicators):
        # Atualiza info com ícones
        self.label_id.setText(f"<b>{indicators['id_coluna']}</b>")
        self.label_rows.setText(f"<b>{indicators['total_linhas']}</b>")
        self.label_cols.setText(f"<b>{indicators['total_colunas']}</b>")

        resumo = []
        append = resumo.append

        append("""
        <div style="font-family: JetBrains Mono, Consolas, monospace; font-size: 13px; color: #FFFFFF;">
        """)

        if not indicators.get('agrupamentos'):
            append("""
            <div style="color:#FF5E5B; margin-top:8px;">
                <b>Aviso:</b> Nenhuma coluna com valores repetidos ou relevantes para agrupamento detectada.
            </div>
            """)
        else:
            for grp in indicators['agrupamentos']:
                append(f"""
                <div style="margin-top:18px; margin-bottom:2px; font-weight:bold; color:#A3A3FF; font-size:15px;">
                    {grp['coluna']} <span style="color:#B9BBBE; font-size:12px;">({grp.get('tipo', '-')})</span>
                </div>
                """)
                if grp.get('estatisticas'):
                    append('<div style="margin-left:18px; color:#43B581; font-size:13px;"><b>Estatísticas:</b></div><ul style="margin:0 0 4px 32px; color:#B9BBBE;">')
                    for k, v in grp['estatisticas'].items():
                        append(f'<li><b>{k.capitalize()}:</b> {v}</li>')
                    append('</ul>')
                if grp.get('tabela') is not None:
                    df = grp['tabela']
                    cols = df.columns[:3]
                    append('<table style="margin-left:18px; background:#23272A; border-collapse:collapse; margin-top:2px; font-size:12px;">')
                    append('<tr>' + ''.join(f'<th style="border-bottom:1px solid #5865F2; color:#F5F5F5; padding:2px 8px;">{col}</th>' for col in cols) + '</tr>')
                    for _, row in df.head(8).iterrows():
                        append('<tr>' + ''.join(f'<td style="padding:1px 8px; color:#B9BBBE;">{row[c]}</td>' for c in cols) + '</tr>')
                    if len(df) > 8:
                        append(f'<tr><td colspan="{len(cols)}" style="color:#AAAAAA; font-style:italic; padding-left:6px;">... e mais {len(df)-8} registros.</td></tr>')
                    append('</table>')
        append("</div>")
        self.output.setHtml(''.join(resumo))

        # Mantém envio para Dash
        try:
            url = 'http://127.0.0.1:8050/update_data'
            headers = {'Content-Type': 'application/json'}
            json_data = prepare_indicators_for_json(indicators)
            requests.post(url, json=json_data, headers=headers, timeout=5)
        except Exception:
            pass
        finally:
            self.progress.setVisible(False)
            self.analyze_btn.setEnabled(True)

    def show_error(self, msg):
        self.output.setPlainText(f"Erro na análise: {msg}")
        self.progress.setVisible(False)
        self.analyze_btn.setEnabled(True)

    def _start_dash(self):
        script = os.path.join(os.path.dirname(__file__), 'app.py')
        if os.path.exists(script):
            try:
                subprocess.Popen([sys.executable, script])
            except Exception as e:
                print(f'Erro iniciando Dash: {e}')


def run_gui():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    run_gui()
