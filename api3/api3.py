from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins='*')

@app.route('/upload-excel', methods=['POST'])
def upload_excel():
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file provided'}), 400
    
    try:
        df = pd.read_excel(file)
        
        # Verificar e lidar com valores de data/hora ausentes
        if df.isnull().values.any():
            # Substituir valores NaN por 0
            df = df.fillna(0)
        
        # Convertendo os dados do DataFrame para um formato JSON serializável
        data = df.to_dict(orient='records')
        return jsonify({'data': data}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002)

