from flask import Flask
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Aplicando CORS para permitir acessos de qualquer origem

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=port)
