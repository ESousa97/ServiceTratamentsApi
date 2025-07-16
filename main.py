#!/usr/bin/env python3
"""
Intelligent CSV Processor - Ponto de entrada principal
Sistema inteligente de processamento e análise de dados CSV com rede neural
"""

import sys
import os
import logging
from pathlib import Path

# Adiciona o diretório raiz ao path para importações
sys.path.insert(0, str(Path(__file__).parent))

def setup_logging():
    """Configura sistema de logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_dependencies():
    """Verifica dependências necessárias"""
    required_packages = [
        'flask', 'pandas', 'numpy', 'scikit-learn', 
        'plotly', 'sentence-transformers', 'torch'
    ]
    import_map = {
        'scikit-learn': 'sklearn',
        'sentence-transformers': 'sentence_transformers',
    }

    missing_packages = []
    for package in required_packages:
        import_name = import_map.get(package, package.replace('-', '_'))
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"❌ Pacotes ausentes: {', '.join(missing_packages)}")
        print("Execute: pip install -r requirements.txt")
        return False

    print("✅ Todas as dependências estão instaladas")
    return True

def create_directories():
    """Cria diretórios necessários"""
    directories = [
        'uploads', 'temp', 'outputs', 'logs', 
        'models/cache', 'static/css', 'static/js'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print("✅ Diretórios criados/verificados")

def initialize_models():
    """Inicializa modelos de ML"""
    try:
        # CORREÇÃO: Import apenas no momento de uso, não no topo
        import importlib
        embeddings_module = importlib.import_module('models.embeddings')
        EmbeddingEngine = getattr(embeddings_module, 'EmbeddingEngine')
        
        # Cria uma instância local
        embedding_engine = EmbeddingEngine()
        model_info = embedding_engine.get_model_info()
        
        print(f"✅ Modelo de embeddings carregado: {model_info['model_name']}")
        print(f"   - Dimensão: {model_info['embedding_dimension']}")
        print(f"   - Device: {model_info['device']}")
        
        return embedding_engine
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelos: {e}")
        return None

def run_tests():
    """Executa testes básicos do sistema"""
    try:
        from config.database import db
        from core.memory_manager import memory_manager

        # Teste do banco de dados
        test_job_id = db.create_job("test.csv", 1024, 100, 10)
        db.update_job_status(test_job_id, 'completed')
        job_info = db.get_job(test_job_id)
        assert job_info is not None
        print("✅ Banco de dados funcionando")

        # Teste de memória
        stats = memory_manager.get_memory_stats()
        assert stats.total_gb > 0
        print(f"✅ Monitoramento de memória: {stats.process_memory_gb:.2f}GB em uso")

        return True
    except Exception as e:
        print(f"❌ Erro nos testes: {e}")
        return False

def print_banner():
    """Exibe banner da aplicação"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║              🧠 INTELLIGENT CSV PROCESSOR 🧠                ║
║                                                              ║
║    Sistema de Análise Automatizada de Dados com IA          ║
║    • Processamento semântico com redes neurais              ║
║    • Análise estatística avançada                           ║
║    • Detecção automática de padrões                         ║
║    • Dashboards interativos gerados automaticamente         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_usage_info(embedding_engine=None):
    """Exibe informações de uso"""
    usage_info = """
🚀 COMO USAR:
  1. Acesse http://localhost:5000 no seu navegador
  2. Faça upload de um arquivo CSV
  3. Aguarde o processamento automático
  4. Explore os dashboards e análises geradas

📊 RECURSOS DISPONÍVEIS:
  • Análise estatística completa
  • Detecção de padrões temporais
  • Análise semântica de texto
  • Visualizações interativas
  • Relatórios em múltiplos formatos

⚙️ CONFIGURAÇÕES:
  • Memória máxima: {max_memory}GB
  • Tamanho máximo de arquivo: {max_file_size}MB
  • Modelo de IA: {model_name}

🔧 APIs DISPONÍVEIS:
  • POST /upload - Upload de arquivo
  • POST /process/<job_id> - Iniciar processamento
  • GET /status/<job_id> - Status do processamento
  • GET /results/<job_id> - Resultados completos
  • GET /dashboard/<job_id> - Dashboard interativo
  • GET /health - Health check do sistema
    """

    try:
        from config.settings import config
        model_name = embedding_engine.get_model_info()['model_name'] if embedding_engine else "Carregando..."
        print(usage_info.format(
            max_memory=config.MAX_MEMORY_GB,
            max_file_size=config.MAX_FILE_SIZE_MB,
            model_name=model_name
        ))
    except:
        print(usage_info.format(
            max_memory="4",
            max_file_size="500",
            model_name="Carregando..."
        ))

def main():
    """Função principal"""
    print_banner()

    print("🔄 Inicializando sistema...")

    # 1. Verificar dependências
    if not check_dependencies():
        sys.exit(1)

    # 2. Criar diretórios
    create_directories()

    # 3. Configurar logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Sistema iniciando...")

    # 4. Inicializar modelos
    embedding_engine = initialize_models()
    if not embedding_engine:
        print("⚠️  Modelos não puderam ser carregados, mas a aplicação continuará")

    # 5. Executar testes
    if not run_tests():
        print("⚠️  Alguns testes falharam, mas a aplicação continuará")

    # 6. Exibir informações de uso
    print_usage_info(embedding_engine)

    # 7. Configurar instância global para o app
    if embedding_engine:
        # Armazena a instância em um local acessível globalmente
        import builtins
        builtins.global_embedding_engine = embedding_engine
        
        # Inicializa componentes semânticos
        try:
            from core.semantic_engine import initialize_semantic_components
            initialize_semantic_components(embedding_engine)
            print("✅ Componentes semânticos inicializados")
        except Exception as e:
            print(f"⚠️  Erro ao inicializar componentes semânticos: {e}")

    # 8. Iniciar servidor
    try:
        print("\n🌟 Sistema pronto! Iniciando servidor...")
        print("   Pressione Ctrl+C para parar\n")

        # CORREÇÃO: Import da aplicação Flask de forma tardia
        import importlib
        app_module = importlib.import_module('api.app')
        app = getattr(app_module, 'app')
        
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,  # Produção
            threaded=True
        )

    except KeyboardInterrupt:
        print("\n\n👋 Encerrando sistema...")
        logger.info("Sistema encerrado pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro fatal: {e}")
        logger.error(f"Erro fatal: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        try:
            from core.memory_manager import memory_manager
            memory_manager.stop_monitoring()
        except:
            pass
        print("✅ Cleanup concluído")

if __name__ == "__main__":
    main()
    