import pandas as pd
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import io
import base64

logger = logging.getLogger(__name__)

class ReportExporter:
    """Exportador de relatórios em múltiplos formatos"""
    
    def __init__(self):
        self.supported_formats = ['json', 'csv', 'xlsx', 'html', 'pdf']
    
    def export_analysis_report(self, analysis_results: Dict[str, Any], 
                             format_type: str = 'json',
                             filename: str = None) -> Dict[str, Any]:
        """Exporta relatório de análise no formato especificado"""
        try:
            if format_type not in self.supported_formats:
                raise ValueError(f"Formato não suportado: {format_type}")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if not filename:
                filename = f"analysis_report_{timestamp}"
            
            # Remove extensão se já estiver presente
            filename = filename.split('.')[0]
            
            if format_type == 'json':
                return self._export_json(analysis_results, filename)
            elif format_type == 'csv':
                return self._export_csv(analysis_results, filename)
            elif format_type == 'xlsx':
                return self._export_xlsx(analysis_results, filename)
            elif format_type == 'html':
                return self._export_html(analysis_results, filename)
            elif format_type == 'pdf':
                return self._export_pdf(analysis_results, filename)
            
        except Exception as e:
            logger.error(f"Erro na exportação: {e}")
            return {'error': str(e)}
    
    def _export_json(self, analysis_results: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Exporta para formato JSON"""
        try:
            # Serializa dados garantindo compatibilidade
            serializable_data = self._make_serializable(analysis_results)
            
            json_content = json.dumps(serializable_data, indent=2, ensure_ascii=False)
            
            return {
                'format': 'json',
                'filename': f"{filename}.json",
                'content': json_content,
                'size': len(json_content.encode('utf-8')),
                'mime_type': 'application/json'
            }
            
        except Exception as e:
            logger.error(f"Erro na exportação JSON: {e}")
            return {'error': str(e)}
    
    def _export_csv(self, analysis_results: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Exporta para formato CSV"""
        try:
            # Extrai dados tabulares dos resultados
            csv_data = self._extract_tabular_data(analysis_results)
            
            if not csv_data:
                return {'error': 'Nenhum dado tabular encontrado para exportação CSV'}
            
            # Cria DataFrame e converte para CSV
            df = pd.DataFrame(csv_data)
            csv_content = df.to_csv(index=False, encoding='utf-8')
            
            return {
                'format': 'csv',
                'filename': f"{filename}.csv",
                'content': csv_content,
                'size': len(csv_content.encode('utf-8')),
                'mime_type': 'text/csv'
            }
            
        except Exception as e:
            logger.error(f"Erro na exportação CSV: {e}")
            return {'error': str(e)}
    
    def _export_xlsx(self, analysis_results: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Exporta para formato Excel"""
        try:
            # Cria buffer em memória
            buffer = io.BytesIO()
            
            # Cria workbook com múltiplas sheets
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # Sheet de resumo
                summary_data = self._create_summary_sheet(analysis_results)
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Resumo', index=False)
                
                # Sheet de dados tabulares
                tabular_data = self._extract_tabular_data(analysis_results)
                if tabular_data:
                    tabular_df = pd.DataFrame(tabular_data)
                    tabular_df.to_excel(writer, sheet_name='Dados', index=False)
                
                # Sheet de estatísticas
                stats_data = self._extract_statistics_data(analysis_results)
                if stats_data:
                    stats_df = pd.DataFrame(stats_data)
                    stats_df.to_excel(writer, sheet_name='Estatísticas', index=False)
            
            buffer.seek(0)
            excel_content = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return {
                'format': 'xlsx',
                'filename': f"{filename}.xlsx",
                'content': excel_content,
                'encoding': 'base64',
                'size': len(buffer.getvalue()),
                'mime_type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            }
            
        except Exception as e:
            logger.error(f"Erro na exportação Excel: {e}")
            return {'error': str(e)}
    
    def _export_html(self, analysis_results: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Exporta para formato HTML"""
        try:
            html_content = self._generate_html_report(analysis_results)
            
            return {
                'format': 'html',
                'filename': f"{filename}.html",
                'content': html_content,
                'size': len(html_content.encode('utf-8')),
                'mime_type': 'text/html'
            }
            
        except Exception as e:
            logger.error(f"Erro na exportação HTML: {e}")
            return {'error': str(e)}
    
    def _export_pdf(self, analysis_results: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Exporta para formato PDF (simplificado)"""
        try:
            # Para PDF, criamos um HTML e retornamos instruções
            html_content = self._generate_html_report(analysis_results)
            
            # Nota: Para conversão real PDF seria necessário biblioteca como weasyprint
            return {
                'format': 'pdf',
                'filename': f"{filename}.pdf",
                'content': html_content,
                'note': 'Conversão PDF não implementada - use HTML como base',
                'mime_type': 'text/html'
            }
            
        except Exception as e:
            logger.error(f"Erro na exportação PDF: {e}")
            return {'error': str(e)}
    
    def _make_serializable(self, obj: Any) -> Any:
        """Converte objeto para formato serializável JSON"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict() if hasattr(obj, 'to_dict') else str(obj)
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif callable(obj):
            return str(obj)
        else:
            return obj
    
    def _extract_tabular_data(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extrai dados em formato tabular dos resultados"""
        tabular_data = []
        
        try:
            # Extrai dados estatísticos
            if 'statistical_analysis' in analysis_results:
                stats = analysis_results['statistical_analysis']
                if 'basic_stats' in stats:
                    basic_stats = stats['basic_stats']
                    tabular_data.append({
                        'Métrica': 'Linhas',
                        'Valor': basic_stats.get('shape', [0, 0])[0],
                        'Categoria': 'Dimensões'
                    })
                    tabular_data.append({
                        'Métrica': 'Colunas',
                        'Valor': basic_stats.get('shape', [0, 0])[1],
                        'Categoria': 'Dimensões'
                    })
                    tabular_data.append({
                        'Métrica': 'Uso de Memória (MB)',
                        'Valor': round(basic_stats.get('memory_usage_mb', 0), 2),
                        'Categoria': 'Performance'
                    })
            
            # Extrai dados de qualidade
            if 'data_quality' in analysis_results.get('statistical_analysis', {}):
                quality = analysis_results['statistical_analysis']['data_quality']
                tabular_data.append({
                    'Métrica': 'Score de Qualidade',
                    'Valor': round(quality.get('overall_score', 0) * 100, 1),
                    'Categoria': 'Qualidade'
                })
        
        except Exception as e:
            logger.warning(f"Erro ao extrair dados tabulares: {e}")
        
        return tabular_data
    
    def _extract_statistics_data(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extrai dados estatísticos detalhados"""
        stats_data = []
        
        try:
            if 'statistical_analysis' in analysis_results:
                distributions = analysis_results['statistical_analysis'].get('distributions', {})
                
                for column, dist_info in distributions.items():
                    stats_data.append({
                        'Coluna': column,
                        'Assimetria': round(dist_info.get('skewness', 0), 3),
                        'Curtose': round(dist_info.get('kurtosis', 0), 3),
                        'Tipo_Distribuição': dist_info.get('distribution_type', 'unknown'),
                        'É_Normal': dist_info.get('normality_test', {}).get('is_normal', False)
                    })
        
        except Exception as e:
            logger.warning(f"Erro ao extrair dados estatísticos: {e}")
        
        return stats_data
    
    def _create_summary_sheet(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Cria dados para sheet de resumo"""
        summary_data = []
        
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            summary_data.append({
                'Item': 'Data do Relatório',
                'Valor': timestamp,
                'Descrição': 'Timestamp da geração do relatório'
            })
            
            # Adiciona informações básicas se disponíveis
            if 'statistical_analysis' in analysis_results:
                basic_stats = analysis_results['statistical_analysis'].get('basic_stats', {})
                shape = basic_stats.get('shape', [0, 0])
                
                summary_data.extend([
                    {
                        'Item': 'Total de Registros',
                        'Valor': shape[0],
                        'Descrição': 'Número total de linhas no dataset'
                    },
                    {
                        'Item': 'Total de Colunas',
                        'Valor': shape[1],
                        'Descrição': 'Número total de colunas no dataset'
                    },
                    {
                        'Item': 'Uso de Memória (MB)',
                        'Valor': round(basic_stats.get('memory_usage_mb', 0), 2),
                        'Descrição': 'Memória utilizada pelo dataset'
                    }
                ])
        
        except Exception as e:
            logger.warning(f"Erro ao criar resumo: {e}")
        
        return summary_data
    
    def _generate_html_report(self, analysis_results: Dict[str, Any]) -> str:
        """Gera relatório em formato HTML"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            html_template = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório de Análise de Dados</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            line-height: 1.6;
            color: #333;
        }}
        .header {{
            text-align: center;
            border-bottom: 2px solid #007bff;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .section {{
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: #f8f9fa;
            border-left: 4px solid #007bff;
            padding: 15px;
            margin: 10px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #007bff;
            color: white;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Relatório de Análise de Dados</h1>
        <p class="timestamp">Gerado em: {timestamp}</p>
    </div>
    
    <div class="section">
        <h2>📊 Resumo Executivo</h2>
        {self._generate_summary_section(analysis_results)}
    </div>
    
    <div class="section">
        <h2>📈 Análise Estatística</h2>
        {self._generate_stats_section(analysis_results)}
    </div>
    
    <div class="section">
        <h2>🔍 Qualidade dos Dados</h2>
        {self._generate_quality_section(analysis_results)}
    </div>
    
    <div class="section">
        <h2>🎯 Recomendações</h2>
        {self._generate_recommendations_section(analysis_results)}
    </div>
</body>
</html>
            """
            
            return html_template
            
        except Exception as e:
            logger.error(f"Erro ao gerar HTML: {e}")
            return f"<html><body><h1>Erro na geração do relatório</h1><p>{e}</p></body></html>"
    
    def _generate_summary_section(self, analysis_results: Dict[str, Any]) -> str:
        """Gera seção de resumo do HTML"""
        try:
            if 'statistical_analysis' not in analysis_results:
                return "<p>Dados de resumo não disponíveis.</p>"
            
            basic_stats = analysis_results['statistical_analysis'].get('basic_stats', {})
            shape = basic_stats.get('shape', [0, 0])
            memory_mb = basic_stats.get('memory_usage_mb', 0)
            
            return f"""
            <div class="metric-card">
                <strong>Dimensões do Dataset:</strong> {shape[0]:,} registros × {shape[1]} colunas
            </div>
            <div class="metric-card">
                <strong>Uso de Memória:</strong> {memory_mb:.1f} MB
            </div>
            <div class="metric-card">
                <strong>Colunas Numéricas:</strong> {basic_stats.get('numeric_columns', 0)}
            </div>
            <div class="metric-card">
                <strong>Colunas Categóricas:</strong> {basic_stats.get('categorical_columns', 0)}
            </div>
            """
        except:
            return "<p>Erro ao gerar resumo.</p>"
    
    def _generate_stats_section(self, analysis_results: Dict[str, Any]) -> str:
        """Gera seção de estatísticas do HTML"""
        try:
            if 'statistical_analysis' not in analysis_results:
                return "<p>Dados estatísticos não disponíveis.</p>"
            
            return "<p>Análise estatística processada com sucesso. Verifique dados detalhados em outras seções.</p>"
        except:
            return "<p>Erro ao gerar estatísticas.</p>"
    
    def _generate_quality_section(self, analysis_results: Dict[str, Any]) -> str:
        """Gera seção de qualidade do HTML"""
        try:
            quality_info = analysis_results.get('statistical_analysis', {}).get('data_quality', {})
            
            if not quality_info:
                return "<p>Informações de qualidade não disponíveis.</p>"
            
            score = quality_info.get('overall_score', 0) * 100
            level = quality_info.get('quality_level', 'unknown')
            
            return f"""
            <div class="metric-card">
                <strong>Score de Qualidade:</strong> {score:.1f}%
            </div>
            <div class="metric-card">
                <strong>Nível de Qualidade:</strong> {level}
            </div>
            """
        except:
            return "<p>Erro ao gerar informações de qualidade.</p>"
    
    def _generate_recommendations_section(self, analysis_results: Dict[str, Any]) -> str:
        """Gera seção de recomendações do HTML"""
        try:
            recommendations = analysis_results.get('statistical_analysis', {}).get('data_quality', {}).get('recommendations', [])
            
            if not recommendations:
                return "<p>Nenhuma recomendação específica identificada.</p>"
            
            rec_html = "<ul>"
            for rec in recommendations:
                rec_html += f"<li>{rec}</li>"
            rec_html += "</ul>"
            
            return rec_html
        except:
            return "<p>Erro ao gerar recomendações.</p>"

# Instância global
report_exporter = ReportExporter()
