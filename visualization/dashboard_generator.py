import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import json
from core.memory_manager import memory_manager

logger = logging.getLogger(__name__)

class DashboardGenerator:
    """Gerador automático de dashboards interativos - VERSÃO CORRIGIDA"""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
        self.template = "plotly_white"
        
    @memory_manager.memory_limiter
    def generate_complete_dashboard(self, df: pd.DataFrame, 
                                  analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Gera dashboard completo - VERSÃO CORRIGIDA"""
        logger.info("Gerando dashboard completo")
        
        # Estrutura mais simples e funcional
        dashboard = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'dataset_size_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
            },
            'charts': self._generate_essential_charts(df),
            'summary': self._generate_summary_cards(df, analysis_results),
            'data_preview': self._generate_data_preview(df)
        }
        
        return dashboard
    
    def _generate_essential_charts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Gera gráficos essenciais de forma mais simples"""
        charts = []
        
        try:
            # 1. Gráfico de tipos de dados
            type_chart = self._create_simple_data_types_chart(df)
            if type_chart:
                charts.append(type_chart)
            
            # 2. Gráfico de completude
            completeness_chart = self._create_simple_completeness_chart(df)
            if completeness_chart:
                charts.append(completeness_chart)
            
            # 3. Histogramas para colunas numéricas (máximo 4)
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:4]
            for col in numeric_cols:
                hist_chart = self._create_simple_histogram(df, col)
                if hist_chart:
                    charts.append(hist_chart)
            
            # 4. Gráficos de barras para colunas categóricas (máximo 3)
            categorical_cols = df.select_dtypes(include=['object']).columns[:3]
            for col in categorical_cols:
                if df[col].nunique() <= 20:  # Só se tiver até 20 categorias
                    bar_chart = self._create_simple_bar_chart(df, col)
                    if bar_chart:
                        charts.append(bar_chart)
            
        except Exception as e:
            logger.error(f"Erro ao gerar gráficos: {e}")
            charts.append({
                'title': 'Erro na Geração de Gráficos',
                'type': 'error',
                'message': str(e)
            })
        
        return charts
    
    def _create_simple_data_types_chart(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Cria gráfico simples de tipos de dados"""
        try:
            type_counts = {}
            
            for column in df.columns:
                dtype = str(df[column].dtype)
                if 'int' in dtype or 'float' in dtype:
                    type_name = 'Numérico'
                elif 'object' in dtype:
                    type_name = 'Texto'
                elif 'datetime' in dtype:
                    type_name = 'Data/Hora'
                elif 'bool' in dtype:
                    type_name = 'Booleano'
                else:
                    type_name = 'Outro'
                
                type_counts[type_name] = type_counts.get(type_name, 0) + 1
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=list(type_counts.keys()),
                    values=list(type_counts.values()),
                    hole=0.3,
                    textinfo='label+percent'
                )
            ])
            
            fig.update_layout(
                title="Distribuição de Tipos de Dados",
                template=self.template,
                height=400,
                showlegend=True
            )
            
            return {
                'title': 'Tipos de Dados',
                'type': 'pie',
                'chart_json': fig.to_json(),
                'description': f"Distribuição dos {len(df.columns)} campos por tipo de dado"
            }
            
        except Exception as e:
            logger.error(f"Erro no gráfico de tipos: {e}")
            return None
    
    def _create_simple_completeness_chart(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Cria gráfico simples de completude"""
        try:
            completeness_data = []
            
            for column in df.columns[:15]:  # Máximo 15 colunas
                null_count = df[column].isnull().sum()
                completeness = ((len(df) - null_count) / len(df)) * 100
                
                completeness_data.append({
                    'column': column,
                    'completeness': round(completeness, 1)
                })
            
            completeness_df = pd.DataFrame(completeness_data)
            completeness_df = completeness_df.sort_values('completeness', ascending=True)
            
            colors = ['#e74c3c' if x < 80 else '#f39c12' if x < 95 else '#2ecc71' 
                     for x in completeness_df['completeness']]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=completeness_df['completeness'],
                y=completeness_df['column'],
                orientation='h',
                marker_color=colors,
                text=completeness_df['completeness'].astype(str) + '%',
                textposition='inside'
            ))
            
            fig.update_layout(
                title="Completude dos Dados (%)",
                xaxis_title="Completude (%)",
                yaxis_title="Colunas",
                template=self.template,
                height=max(400, len(completeness_df) * 25)
            )
            
            return {
                'title': 'Completude dos Dados',
                'type': 'horizontal_bar',
                'chart_json': fig.to_json(),
                'description': "Percentual de dados completos por coluna"
            }
            
        except Exception as e:
            logger.error(f"Erro no gráfico de completude: {e}")
            return None
    
    def _create_simple_histogram(self, df: pd.DataFrame, column: str) -> Optional[Dict[str, Any]]:
        """Cria histograma simples"""
        try:
            data = df[column].dropna()
            
            if len(data) == 0:
                return None
            
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=data,
                nbinsx=30,
                marker_color='#3498db',
                opacity=0.7
            ))
            
            # Adiciona linha de média
            mean_val = data.mean()
            fig.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Média: {mean_val:.2f}"
            )
            
            fig.update_layout(
                title=f"Distribuição: {column}",
                xaxis_title=column,
                yaxis_title="Frequência",
                template=self.template,
                height=400
            )
            
            return {
                'title': f'Histograma: {column}',
                'type': 'histogram',
                'chart_json': fig.to_json(),
                'description': f"Distribuição da coluna {column}",
                'stats': {
                    'mean': round(float(mean_val), 2),
                    'median': round(float(data.median()), 2),
                    'std': round(float(data.std()), 2),
                    'min': round(float(data.min()), 2),
                    'max': round(float(data.max()), 2)
                }
            }
            
        except Exception as e:
            logger.error(f"Erro no histograma de {column}: {e}")
            return None
    
    def _create_simple_bar_chart(self, df: pd.DataFrame, column: str) -> Optional[Dict[str, Any]]:
        """Cria gráfico de barras simples"""
        try:
            value_counts = df[column].value_counts()
            
            # Limita a 15 categorias
            if len(value_counts) > 15:
                top_values = value_counts.head(14)
                others_sum = value_counts.tail(len(value_counts) - 14).sum()
                top_values['Outros'] = others_sum
                value_counts = top_values
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=value_counts.index.astype(str),
                y=value_counts.values,
                marker_color='#2ecc71',
                text=value_counts.values,
                textposition='outside'
            ))
            
            fig.update_layout(
                title=f"Distribuição: {column}",
                xaxis_title=column,
                yaxis_title="Frequência",
                template=self.template,
                height=400,
                xaxis_tickangle=-45
            )
            
            return {
                'title': f'Frequências: {column}',
                'type': 'bar',
                'chart_json': fig.to_json(),
                'description': f"Frequência das categorias de {column}"
            }
            
        except Exception as e:
            logger.error(f"Erro no gráfico de barras de {column}: {e}")
            return None
    
    def _generate_summary_cards(self, df: pd.DataFrame, 
                              analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gera cards de resumo"""
        cards = []
        
        try:
            # Card 1: Dimensões
            cards.append({
                'title': 'Dimensões',
                'value': f"{len(df):,} × {len(df.columns)}",
                'subtitle': 'Linhas × Colunas',
                'icon': 'table',
                'color': '#3498db'
            })
            
            # Card 2: Completude
            total_cells = len(df) * len(df.columns)
            non_null_cells = df.count().sum()
            completeness = (non_null_cells / total_cells) * 100 if total_cells > 0 else 0
            
            cards.append({
                'title': 'Completude',
                'value': f"{completeness:.1f}%",
                'subtitle': 'Dados completos',
                'icon': 'check-circle',
                'color': '#2ecc71' if completeness > 80 else '#f39c12'
            })
            
            # Card 3: Tipos de dados
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            text_cols = len(df.select_dtypes(include=['object']).columns)
            
            cards.append({
                'title': 'Tipos',
                'value': f"{numeric_cols}N + {text_cols}T",
                'subtitle': 'Numérico + Texto',
                'icon': 'layers',
                'color': '#9b59b6'
            })
            
            # Card 4: Memória
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            cards.append({
                'title': 'Memória',
                'value': f"{memory_mb:.1f} MB",
                'subtitle': 'Uso de RAM',
                'icon': 'memory',
                'color': '#e67e22'
            })
            
        except Exception as e:
            logger.error(f"Erro ao gerar cards: {e}")
            cards.append({
                'title': 'Erro',
                'value': 'N/A',
                'subtitle': str(e),
                'icon': 'alert',
                'color': '#e74c3c'
            })
        
        return cards
    
    def _generate_data_preview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Gera preview dos dados"""
        try:
            preview_df = df.head(10)
            
            return {
                'columns': df.columns.tolist(),
                'data': preview_df.to_dict('records'),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'total_rows': len(df),
                'preview_rows': len(preview_df)
            }
            
        except Exception as e:
            logger.error(f"Erro no preview: {e}")
            return {
                'error': str(e),
                'columns': [],
                'data': [],
                'total_rows': 0
            }

# Instância global
dashboard_generator = DashboardGenerator()
