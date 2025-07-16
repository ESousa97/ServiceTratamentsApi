from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
from flask_cors import CORS
import os
import threading
import time
from datetime import datetime
import logging
from pathlib import Path
import traceback

# Importações dos módulos do projeto
from config.settings import config, ConfigManager
from config.database import db, cache
from core.file_handler import CSVProcessor, FileValidator
from core.memory_manager import memory_manager
from core.semantic_engine import semantic_processor
# CORREÇÃO: Removido import que causava problema circular
# from models.embeddings import embedding_engine
from analysis.statistical_analyzer import statistical_analyzer
from analysis.pattern_detector import pattern_detector
from visualization.dashboard_generator import dashboard_generator

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_embedding_engine():
    """Obtém a instância global do embedding engine"""
    try:
        import builtins
        return getattr(builtins, 'global_embedding_engine', None)
    except:
        # Fallback: cria nova instância se necessário
        from models.embeddings import EmbeddingEngine
        return EmbeddingEngine()

class IntelligentCSVProcessor:
    """Processador principal da aplicação"""
    
    def __init__(self):
        self.csv_processor = CSVProcessor()
        self.file_validator = FileValidator()
        self.current_jobs = {}
        
    def estimate_processing_time(self, metadata) -> int:
        """Estima tempo de processamento em minutos"""
        # Fórmula baseada no tamanho e complexidade dos dados
        rows = metadata.rows_count
        cols = metadata.columns_count
        memory_mb = metadata.estimated_memory_mb
        
        # Fatores de complexidade
        base_time = (rows * cols) / 100000  # Base: 1 minuto para cada 100k células
        memory_factor = memory_mb / 1000     # Fator adicional por GB
        complexity_factor = 1.5 if cols > 50 else 1.0  # Mais colunas = mais complexo
        
        estimated_minutes = base_time + memory_factor + complexity_factor
        
        # Limites mínimo e máximo
        return max(1, min(120, int(estimated_minutes)))
    
    def process_file_async(self, file_path: str, job_id: int):
        """Processa arquivo de forma assíncrona"""
        try:
            # Atualiza status para processando
            db.update_job_status(job_id, 'processing', progress=0.0)
            
            logger.info(f"Iniciando processamento do job {job_id}")
            
            # Etapa 1: Análise do arquivo (10%)
            metadata = self.csv_processor.analyze_file(file_path)
            db.update_job_status(job_id, 'processing', progress=10.0)
            
            # Etapa 2: Carregamento dos dados (20%)
            df = self.csv_processor.load_data_chunked(file_path, metadata)
            db.update_job_status(job_id, 'processing', progress=20.0)
            
            # Etapa 3: Análise estatística (40%)
            statistical_results = statistical_analyzer.analyze_dataset(df)
            db.save_analysis(job_id, 'statistical', statistical_results)
            db.update_job_status(job_id, 'processing', progress=40.0)
            
            # Etapa 4: Detecção de padrões (60%)
            pattern_results = pattern_detector.detect_all_patterns(df)
            db.save_analysis(job_id, 'patterns', pattern_results)
            db.update_job_status(job_id, 'processing', progress=60.0)
            
            # Etapa 5: Análise semântica (80%)
            semantic_results = semantic_processor.process_dataframe_semantics(df)
            db.save_analysis(job_id, 'semantic', semantic_results)
            db.update_job_status(job_id, 'processing', progress=80.0)
            
            # Etapa 6: Geração de dashboards (100%)
            all_results = {
                'statistical_analysis': statistical_results,
                'pattern_analysis': pattern_results,
                'semantic_analysis': semantic_results
            }
            
            dashboard_data = dashboard_generator.generate_complete_dashboard(df, all_results)
            db.save_analysis(job_id, 'dashboard', dashboard_data)
            
            # Finaliza processamento
            db.update_job_status(job_id, 'completed', progress=100.0)
            
            logger.info(f"Job {job_id} processado com sucesso")
            
            # Remove da lista de jobs ativos
            if job_id in self.current_jobs:
                del self.current_jobs[job_id]
                
        except Exception as e:
            logger.error(f"Erro no processamento do job {job_id}: {e}")
            logger.error(traceback.format_exc())
            db.update_job_status(job_id, 'failed', error_message=str(e))
            
            # Remove da lista de jobs ativos
            if job_id in self.current_jobs:
                del self.current_jobs[job_id]

# Inicialização da aplicação (DEVE ESTAR NO TOPO, ANTES DAS ROTAS)
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app, origins=config.CORS_ORIGINS)

# Configuração da aplicação
app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = config.MAX_FILE_SIZE_MB * 1024 * 1024

# Cria diretórios necessários
ConfigManager.create_directories(config)

# Inicializa processador principal
processor = IntelligentCSVProcessor()

# Inicia monitoramento de memória
memory_manager.start_monitoring()

# ============================================================================
# ROTAS DA INTERFACE WEB
# ============================================================================

@app.route('/')
def index():
    """Página inicial da aplicação"""
    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve arquivos estáticos"""
    return send_from_directory('static', filename)

@app.route('/health-check')
def health_check_web():
    """Health check para a interface web"""
    embedding_engine = get_embedding_engine()
    model_info = embedding_engine.get_model_info() if embedding_engine else {'model_name': 'Não carregado'}
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'memory_usage_gb': memory_manager.get_memory_stats().process_memory_gb,
        'embedding_model': model_info.get('model_name', 'Não disponível'),
        'config': {
            'max_memory_gb': config.MAX_MEMORY_GB,
            'max_file_size_mb': config.MAX_FILE_SIZE_MB
        }
    })

# ============================================================================
# ROTAS DA API (EXISTENTES)
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de health check"""
    embedding_engine = get_embedding_engine()
    model_info = embedding_engine.get_model_info() if embedding_engine else {'model_name': 'Não carregado'}
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'memory_usage_gb': memory_manager.get_memory_stats().process_memory_gb,
        'embedding_model': model_info.get('model_name', 'Não disponível'),
        'config': {
            'max_memory_gb': config.MAX_MEMORY_GB,
            'max_file_size_mb': config.MAX_FILE_SIZE_MB
        }
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Endpoint para upload de arquivo CSV"""
    try:
        # Verifica se arquivo foi enviado
        if 'file' not in request.files:
            return jsonify({'error': 'Nenhum arquivo enviado'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
        
        # Verifica extensão do arquivo
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in config.ALLOWED_EXTENSIONS:
            return jsonify({'error': f'Extensão não permitida: {file_ext}'}), 400
        
        # Salva arquivo temporariamente
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        file_path = os.path.join(config.UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Analisa arquivo
        try:
            metadata = processor.csv_processor.analyze_file(file_path)
        except Exception as e:
            os.remove(file_path)  # Remove arquivo em caso de erro
            return jsonify({'error': f'Erro na análise do arquivo: {str(e)}'}), 400
        
        # Valida arquivo
        validation_issues = processor.file_validator.validate_file(file_path, metadata)
        if validation_issues:
            os.remove(file_path)  # Remove arquivo em caso de erro
            return jsonify({
                'error': 'Arquivo não passou na validação',
                'issues': validation_issues
            }), 400
        
        # Cria job no banco de dados
        job_id = db.create_job(
            filename=filename,
            file_size=metadata.file_size,
            rows_count=metadata.rows_count,
            columns_count=metadata.columns_count
        )
        
        # Estima tempo de processamento
        estimated_time = processor.estimate_processing_time(metadata)
        
        return jsonify({
            'job_id': job_id,
            'filename': filename,
            'metadata': {
                'rows_count': metadata.rows_count,
                'columns_count': metadata.columns_count,
                'file_size_mb': metadata.file_size / (1024 * 1024),
                'estimated_memory_mb': metadata.estimated_memory_mb,
                'column_names': metadata.column_names[:10],  # Primeiras 10 colunas
                'sample_data': metadata.sample_data
            },
            'estimated_processing_time_minutes': estimated_time,
            'message': f'Arquivo carregado com sucesso. Processamento estimado: {estimated_time} minutos.'
        })
        
    except Exception as e:
        logger.error(f"Erro no upload: {e}")
        return jsonify({'error': f'Erro interno: {str(e)}'}), 500

@app.route('/process/<int:job_id>', methods=['POST'])
def start_processing(job_id):
    """Inicia processamento de um job"""
    try:
        # Verifica se job existe
        job = db.get_job(job_id)
        if not job:
            return jsonify({'error': 'Job não encontrado'}), 404
        
        # Verifica se job já está sendo processado
        if job['status'] in ['processing', 'completed']:
            return jsonify({
                'message': f'Job já está {job["status"]}',
                'job_id': job_id,
                'status': job['status']
            })
        
        # Verifica se arquivo ainda existe
        file_path = os.path.join(config.UPLOAD_FOLDER, job['filename'])
        if not os.path.exists(file_path):
            return jsonify({'error': 'Arquivo não encontrado'}), 404
        
        # Inicia processamento assíncrono
        thread = threading.Thread(
            target=processor.process_file_async,
            args=(file_path, job_id)
        )
        thread.daemon = True
        thread.start()
        
        # Adiciona à lista de jobs ativos
        processor.current_jobs[job_id] = {
            'thread': thread,
            'started_at': datetime.now()
        }
        
        return jsonify({
            'message': 'Processamento iniciado',
            'job_id': job_id,
            'status': 'processing'
        })
        
    except Exception as e:
        logger.error(f"Erro ao iniciar processamento: {e}")
        return jsonify({'error': f'Erro interno: {str(e)}'}), 500

@app.route('/status/<int:job_id>', methods=['GET'])
def get_job_status(job_id):
    """Retorna status de um job"""
    try:
        job = db.get_job(job_id)
        if not job:
            return jsonify({'error': 'Job não encontrado'}), 404
        
        return jsonify({
            'job_id': job_id,
            'status': job['status'],
            'progress': job['progress'],
            'created_at': job['created_at'],
            'started_at': job['started_at'],
            'completed_at': job['completed_at'],
            'error_message': job['error_message'],
            'estimated_duration': job['estimated_duration'],
            'is_active': job_id in processor.current_jobs
        })
        
    except Exception as e:
        logger.error(f"Erro ao obter status: {e}")
        return jsonify({'error': f'Erro interno: {str(e)}'}), 500

@app.route('/results/<int:job_id>', methods=['GET'])
def get_job_results(job_id):
    """Retorna resultados de um job"""
    try:
        job = db.get_job(job_id)
        if not job:
            return jsonify({'error': 'Job não encontrado'}), 404
        
        if job['status'] != 'completed':
            return jsonify({
                'error': 'Job ainda não foi completado',
                'status': job['status']
            }), 400
        
        # Recupera todas as análises
        analyses = db.get_analyses(job_id)
        
        results = {
            'job_info': job,
            'analyses': {}
        }
        
        for analysis in analyses:
            results['analyses'][analysis['analysis_type']] = analysis['results']
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Erro ao obter resultados: {e}")
        return jsonify({'error': f'Erro interno: {str(e)}'}), 500

@app.route('/dashboard/<int:job_id>', methods=['GET'])
def get_dashboard_data(job_id):
    """Retorna dados do dashboard"""
    try:
        job = db.get_job(job_id)
        if not job:
            return jsonify({'error': 'Job não encontrado'}), 404
        
        if job['status'] != 'completed':
            return jsonify({
                'error': 'Dashboard ainda não está disponível',
                'status': job['status']
            }), 400
        
        # Recupera dados do dashboard
        analyses = db.get_analyses(job_id)
        dashboard_data = None
        
        for analysis in analyses:
            if analysis['analysis_type'] == 'dashboard':
                dashboard_data = analysis['results']
                break
        
        if not dashboard_data:
            return jsonify({'error': 'Dados do dashboard não encontrados'}), 404
        
        return jsonify(dashboard_data)
        
    except Exception as e:
        logger.error(f"Erro ao obter dashboard: {e}")
        return jsonify({'error': f'Erro interno: {str(e)}'}), 500

@app.route('/jobs', methods=['GET'])
def list_jobs():
    """Lista todos os jobs"""
    try:
        # Por simplicidade, vamos retornar jobs recentes
        # Em uma implementação completa, isso seria paginado
        with db.get_cursor() as cursor:
            cursor.execute('''
                SELECT id, filename, status, created_at, progress, rows_count, columns_count
                FROM processing_jobs
                ORDER BY created_at DESC
                LIMIT 50
            ''')
            jobs = [dict(row) for row in cursor.fetchall()]
        
        return jsonify({'jobs': jobs})
        
    except Exception as e:
        logger.error(f"Erro ao listar jobs: {e}")
        return jsonify({'error': f'Erro interno: {str(e)}'}), 500

@app.route('/memory-stats', methods=['GET'])
def get_memory_stats():
    """Retorna estatísticas de memória"""
    try:
        stats = memory_manager.get_memory_stats()
        return jsonify({
            'total_gb': stats.total_gb,
            'available_gb': stats.available_gb,
            'used_gb': stats.used_gb,
            'percent_used': stats.percent_used,
            'process_memory_gb': stats.process_memory_gb,
            'max_allowed_gb': config.MAX_MEMORY_GB,
            'within_limit': stats.process_memory_gb <= config.MAX_MEMORY_GB
        })
    except Exception as e:
        logger.error(f"Erro ao obter estatísticas de memória: {e}")
        return jsonify({'error': f'Erro interno: {str(e)}'}), 500
    
# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(413)
def file_too_large(e):
    """Handler para arquivo muito grande"""
    return jsonify({
        'error': f'Arquivo muito grande. Máximo permitido: {config.MAX_FILE_SIZE_MB}MB'
    }), 413

@app.errorhandler(500)
def internal_server_error(e):
    """Handler para erro interno do servidor"""
    logger.error(f"Erro interno do servidor: {e}")
    return jsonify({'error': 'Erro interno do servidor'}), 500

if __name__ == '__main__':
    logger.info("Iniciando servidor da aplicação")
    logger.info(f"Configuração: {config.__class__.__name__}")
    logger.info(f"Memória máxima: {config.MAX_MEMORY_GB}GB")
    logger.info(f"Tamanho máximo de arquivo: {config.MAX_FILE_SIZE_MB}MB")
    
    try:
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=config.DEBUG,
            threaded=True
        )
    finally:
        # Cleanup
        memory_manager.stop_monitoring()
        logger.info("Servidor finalizado")
        