# main.py
import threading
from gui.main_gui import run_gui
from gui.app import app as dash_app

def _start_dash():
    # use_reloader=False evita que o Dash crie processos extras
    dash_app.run_server(host="127.0.0.1", port=8050,
                        debug=False, use_reloader=False)

if __name__ == "__main__":
    threading.Thread(target=_start_dash, daemon=True).start()
    run_gui()
