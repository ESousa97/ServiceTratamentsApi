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
    """Gerador automático de dashboards interativos"""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
        self.template = "plotly_white"
        
    @memory_manager.memory_limiter
    def generate_complete_dashboard(self, df: pd.DataFrame, 
                                  analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Gera dashboard completo com múltiplas visualizações"""
        logger.info("Gerando dashboard completo")
        
        dashboard = {
            'overview': self._create_overview_section(df, analysis_results),
            'data_quality': self._create_data_quality_section(df, analysis_results),
            'statistical_analysis': self._create_statistical_section(df, analysis_results),
            'pattern_analysis': self._create_pattern_section(df, analysis_results),
            'correlation_analysis': self._create_correlation_section(df, analysis_results),
            'time_series': self._create_time_series_section(df, analysis_results),
            'interactive_filters': self._create_filter_controls(df)
        }
        
        return dashboard
    
    def _create_overview_section(self, df: pd.DataFrame, 
                               analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Cria seção de overview geral"""
        overview = {}
        
        # Métricas principais
        metrics = self._create_key_metrics(df, analysis_results)
        overview['metrics'] = metrics
        
        # Gráfico de distribuição de tipos de dados
        data_types_chart = self._create_data_types_chart(df)
        overview['data_types_chart'] = data_types_chart
        
        # Gráfico de completude dos dados
        completeness_chart = self._create_completeness_chart(df)
        overview['completeness_chart'] = completeness_chart
        
        # Sumário executivo
        executive_summary = self._create_executive_summary(df, analysis_results)
        overview['executive_summary'] = executive_summary
        
        return overview
    
    def _create_key_metrics(self, df: pd.DataFrame, 
                          analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Cria métricas-chave do dataset"""
        metrics = []
        
        # Métrica 1: Total de registros
        metrics.append({
            'title': 'Total de Registros',
            'value': len(df),
            'format': 'number',
            'icon': 'database',
            'color': '#3498db'
        })
        
        # Métrica 2: Total de colunas
        metrics.append({
            'title': 'Total de Colunas',
            'value': len(df.columns),
            'format': 'number',
            'icon': 'columns',
            'color': '#2ecc71'
        })
        
        # Métrica 3: Completude dos dados
        total_cells = len(df) * len(df.columns)
        non_null_cells = df.count().sum()
        completeness = (non_null_cells / total_cells) * 100 if total_cells > 0 else 0
        
        metrics.append({
            'title': 'Completude dos Dados',
            'value': completeness,
            'format': 'percentage',
            'icon': 'check-circle',
            'color': '#f39c12' if completeness < 80 else '#2ecc71'
        })
        
        # Métrica 4: Qualidade geral (se disponível)
        if 'statistical_analysis' in analysis_results:
            quality_info = analysis_results['statistical_analysis'].get('data_quality', {})
            quality_score = quality_info.get('overall_score', 0) * 100
            
            metrics.append({
                'title': 'Score de Qualidade',
                'value': quality_score,
                'format': 'percentage',
                'icon': 'star',
                'color': self._get_quality_color(quality_score)
            })
        
        # Métrica 5: Uso de memória
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        metrics.append({
            'title': 'Uso de Memória',
            'value': memory_mb,
            'format': 'mb',
            'icon': 'memory',
            'color': '#9b59b6'
        })
        
        return metrics
    
    def _get_quality_color(self, score: float) -> str:
        """Retorna cor baseada no score de qualidade"""
        if score >= 90:
            return '#2ecc71'  # Verde
        elif score >= 70:
            return '#f39c12'  # Amarelo
        else:
            return '#e74c3c'  # Vermelho
    
    def _create_data_types_chart(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Cria gráfico de distribuição de tipos de dados"""
        type_counts = {}
        
        for column in df.columns:
            dtype = str(df[column].dtype)
            if 'int' in dtype:
                type_name = 'Integer'
            elif 'float' in dtype:
                type_name = 'Float'
            elif 'object' in dtype:
                type_name = 'Text/Object'
            elif 'datetime' in dtype:
                type_name = 'DateTime'
            elif 'bool' in dtype:
                type_name = 'Boolean'
            else:
                type_name = 'Other'
            
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        fig = go.Figure(data=[
            go.Pie(
                labels=list(type_counts.keys()),
                values=list(type_counts.values()),
                hole=0.4,
                textinfo='label+percent',
                marker=dict(colors=self.color_palette[:len(type_counts)])
            )
        ])
        
        fig.update_layout(
            title="Distribuição de Tipos de Dados",
            template=self.template,
            showlegend=True
        )
        
        return {
            'chart': fig.to_json(),
            'data': type_counts,
            'chart_type': 'pie'
        }
    
    def _create_completeness_chart(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Cria gráfico de completude por coluna"""
        completeness_data = []
        
        for column in df.columns:
            null_count = df[column].isnull().sum()
            completeness = ((len(df) - null_count) / len(df)) * 100
            
            completeness_data.append({
                'column': column,
                'completeness': completeness,
                'missing_count': null_count
            })
        
        completeness_df = pd.DataFrame(completeness_data)
        completeness_df = completeness_df.sort_values('completeness')
        
        fig = go.Figure()
        
        # Adiciona barras com cores baseadas na completude
        colors = ['#e74c3c' if x < 80 else '#f39c12' if x < 95 else '#2ecc71' 
                 for x in completeness_df['completeness']]
        
        fig.add_trace(go.Bar(
            x=completeness_df['completeness'],
            y=completeness_df['column'],
            orientation='h',
            marker_color=colors,
            text=completeness_df['completeness'].round(1).astype(str) + '%',
            textposition='inside'
        ))
        
        fig.update_layout(
            title="Completude dos Dados por Coluna",
            xaxis_title="Completude (%)",
            yaxis_title="Colunas",
            template=self.template,
            height=max(400, len(df.columns) * 30)
        )
        
        return {
            'chart': fig.to_json(),
            'data': completeness_data,
            'chart_type': 'horizontal_bar'
        }
    
    def _create_executive_summary(self, df: pd.DataFrame, 
                                analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Cria sumário executivo"""
        summary = {
            'dataset_size': f"{len(df):,} registros × {len(df.columns)} colunas",
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB",
            'data_types': {
                'numeric': len(df.select_dtypes(include=[np.number]).columns),
                'text': len(df.select_dtypes(include=['object']).columns),
                'datetime': len(df.select_dtypes(include=['datetime64']).columns)
            },
            'key_insights': [],
            'recommendations': []
        }
        
        # Adiciona insights baseados na análise
        if 'statistical_analysis' in analysis_results:
            stats = analysis_results['statistical_analysis']
            
            # Insight sobre qualidade
            quality_score = stats.get('data_quality', {}).get('overall_score', 0)
            if quality_score > 0.9:
                summary['key_insights'].append("Dataset possui excelente qualidade")
            elif quality_score < 0.6:
                summary['key_insights'].append("Dataset requer melhorias na qualidade")
            
            # Insight sobre missing values
            missing_info = stats.get('missing_values', {}).get('overall', {})
            missing_pct = missing_info.get('missing_percentage', 0)
            if missing_pct > 20:
                summary['recommendations'].append("Tratar valores faltantes (>20% dos dados)")
        
        return summary
    
    def _create_data_quality_section(self, df: pd.DataFrame, 
                                   analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Cria seção de qualidade dos dados"""
        quality_section = {}
        
        # Gráfico de valores faltantes
        missing_values_chart = self._create_missing_values_heatmap(df)
        quality_section['missing_values_heatmap'] = missing_values_chart
        
        # Gráfico de outliers
        outliers_chart = self._create_outliers_chart(df, analysis_results)
        quality_section['outliers_chart'] = outliers_chart
        
        # Gráfico de duplicatas
        duplicates_chart = self._create_duplicates_chart(df)
        quality_section['duplicates_chart'] = duplicates_chart
        
        return quality_section
    
    def _create_missing_values_heatmap(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Cria heatmap de valores faltantes"""
        # Limita a 50 colunas para performance
        cols_to_show = df.columns[:50] if len(df.columns) > 50 else df.columns
        sample_df = df[cols_to_show]
        
        # Calcula matriz de valores faltantes
        missing_matrix = sample_df.isnull().astype(int)
        
        # Se há muitas linhas, faz amostragem
        if len(missing_matrix) > 1000:
            missing_matrix = missing_matrix.sample(n=1000, random_state=42)
        
        fig = go.Figure(data=go.Heatmap(
            z=missing_matrix.values,
            x=missing_matrix.columns,
            y=missing_matrix.index,
            colorscale=[[0, '#2ecc71'], [1, '#e74c3c']],
            showscale=True,
            colorbar=dict(title="Missing Values")
        ))
        
        fig.update_layout(
            title="Mapa de Valores Faltantes",
            xaxis_title="Colunas",
            yaxis_title="Registros",
            template=self.template,
            height=600
        )
        
        return {
            'chart': fig.to_json(),
            'chart_type': 'heatmap'
        }
    
    def _create_outliers_chart(self, df: pd.DataFrame, 
                             analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Cria gráfico de outliers"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {'message': 'Nenhuma coluna numérica encontrada para análise de outliers'}
        
        # Limita a 10 colunas para visualização
        cols_to_plot = numeric_cols[:10]
        
        fig = make_subplots(
            rows=1, cols=len(cols_to_plot),
            subplot_titles=cols_to_plot,
            horizontal_spacing=0.05
        )
        
        for i, col in enumerate(cols_to_plot):
            data = df[col].dropna()
            
            if len(data) == 0:
                continue
            
            fig.add_trace(
                go.Box(
                    y=data,
                    name=col,
                    boxpoints='outliers',
                    marker_color=self.color_palette[i % len(self.color_palette)]
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title="Detecção de Outliers por Coluna",
            template=self.template,
            height=400,
            showlegend=False
        )
        
        return {
            'chart': fig.to_json(),
            'chart_type': 'boxplot'
        }
    
    def _create_duplicates_chart(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Cria gráfico de análise de duplicatas"""
        duplicate_count = df.duplicated().sum()
        unique_count = len(df) - duplicate_count
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Registros Únicos', 'Registros Duplicados'],
                y=[unique_count, duplicate_count],
                marker_color=['#2ecc71', '#e74c3c'],
                text=[f'{unique_count:,}', f'{duplicate_count:,}'],
                textposition='inside'
            )
        ])
        
        fig.update_layout(
            title="Análise de Duplicatas",
            yaxis_title="Quantidade de Registros",
            template=self.template
        )
        
        return {
            'chart': fig.to_json(),
            'data': {
                'unique_records': unique_count,
                'duplicate_records': duplicate_count,
                'duplicate_percentage': (duplicate_count / len(df)) * 100
            },
            'chart_type': 'bar'
        }
    
    def _create_statistical_section(self, df: pd.DataFrame, 
                                  analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Cria seção de análise estatística"""
        statistical_section = {}
        
        # Gráficos de distribuição
        distribution_charts = self._create_distribution_charts(df)
        statistical_section['distributions'] = distribution_charts
        
        # Matriz de correlação
        correlation_matrix = self._create_correlation_matrix(df)
        statistical_section['correlation_matrix'] = correlation_matrix
        
        # Estatísticas descritivas
        descriptive_stats = self._create_descriptive_stats_table(df)
        statistical_section['descriptive_stats'] = descriptive_stats
        
        return statistical_section
    
    def _create_distribution_charts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Cria gráficos de distribuição para colunas numéricas"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        distribution_charts = []
        
        # Limita a 6 colunas para performance
        cols_to_plot = numeric_cols[:6]
        
        for col in cols_to_plot:
            data = df[col].dropna()
            
            if len(data) == 0:
                continue
            
            # Histograma
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=data,
                nbinsx=30,
                name=col,
                marker_color=self.color_palette[0],
                opacity=0.7
            ))
            
            # Adiciona linha de média
            fig.add_vline(
                x=data.mean(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Média: {data.mean():.2f}"
            )
            
            fig.update_layout(
                title=f"Distribuição de {col}",
                xaxis_title=col,
                yaxis_title="Frequência",
                template=self.template
            )
            
            distribution_charts.append({
                'column': col,
                'chart': fig.to_json(),
                'stats': {
                    'mean': float(data.mean()),
                    'median': float(data.median()),
                    'std': float(data.std()),
                    'min': float(data.min()),
                    'max': float(data.max())
                }
            })
        
        return distribution_charts
    
    def _create_correlation_matrix(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Cria matriz de correlação"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {'message': 'Insuficientes colunas numéricas para matriz de correlação'}
        
        # Limita a 20 colunas para performance
        cols_to_corr = numeric_cols[:20]
        corr_matrix = df[cols_to_corr].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Correlação")
        ))
        
        fig.update_layout(
            title="Matriz de Correlação",
            template=self.template,
            height=600
        )
        
        return {
            'chart': fig.to_json(),
            'data': corr_matrix.to_dict(),
            'chart_type': 'heatmap'
        }
    
    def _create_descriptive_stats_table(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Cria tabela de estatísticas descritivas"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {'message': 'Nenhuma coluna numérica encontrada'}
        
        stats_df = df[numeric_cols].describe()
        
        return {
            'data': stats_df.round(2).to_dict(),
            'columns': stats_df.columns.tolist(),
            'index': stats_df.index.tolist()
        }
    
    def _create_pattern_section(self, df: pd.DataFrame, 
                              analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Cria seção de análise de padrões"""
        pattern_section = {}
        
        # Visualização de clusters (se disponível)
        if 'pattern_analysis' in analysis_results:
            clustering_viz = self._create_clustering_visualization(df, analysis_results)
            pattern_section['clustering'] = clustering_viz
        
        # Análise de padrões temporais
        temporal_viz = self._create_temporal_patterns_viz(df, analysis_results)
        pattern_section['temporal_patterns'] = temporal_viz
        
        return pattern_section
    
    def _create_clustering_visualization(self, df: pd.DataFrame, 
                                       analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Cria visualização de clustering"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {'message': 'Insuficientes colunas numéricas para visualização de clustering'}
        
        # Usa PCA para reduzir dimensionalidade se necessário
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        data = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        if len(numeric_cols) > 2:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(scaled_data)
            x_col, y_col = 'PC1', 'PC2'
            plot_data = pd.DataFrame(pca_data, columns=[x_col, y_col])
        else:
            x_col, y_col = numeric_cols[0], numeric_cols[1]
            plot_data = data[[x_col, y_col]]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=plot_data[x_col],
            y=plot_data[y_col],
            mode='markers',
            marker=dict(
                size=8,
                color=self.color_palette[0],
                opacity=0.6
            ),
            name='Dados'
        ))
        
        fig.update_layout(
            title="Visualização de Dados (Clustering)",
            xaxis_title=x_col,
            yaxis_title=y_col,
            template=self.template
        )
        
        return {
            'chart': fig.to_json(),
            'chart_type': 'scatter'
        }
    
    def _create_temporal_patterns_viz(self, df: pd.DataFrame, 
                                    analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Cria visualização de padrões temporais"""
        # Procura por colunas de data
        date_cols = []
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]':
                date_cols.append(col)
            elif df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].head(10), errors='raise')
                    date_cols.append(col)
                except:
                    continue
        
        if not date_cols:
            return {'message': 'Nenhuma coluna temporal encontrada'}
        
        # Usa primeira coluna de data encontrada
        date_col = date_cols[0]
        
        try:
            # Converte para datetime se necessário
            if df[date_col].dtype != 'datetime64[ns]':
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Cria série temporal de contagem
            date_counts = df[date_col].dt.date.value_counts().sort_index()
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=date_counts.index,
                y=date_counts.values,
                mode='lines+markers',
                name='Frequência',
                line=dict(color=self.color_palette[0])
            ))
            
            fig.update_layout(
                title=f"Padrão Temporal - {date_col}",
                xaxis_title="Data",
                yaxis_title="Frequência",
                template=self.template
            )
            
            return {
                'chart': fig.to_json(),
                'chart_type': 'time_series'
            }
            
        except Exception as e:
            return {'message': f'Erro na análise temporal: {str(e)}'}
    
    def _create_correlation_section(self, df: pd.DataFrame, 
                                  analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Cria seção de análise de correlação"""
        return self._create_correlation_matrix(df)
    
    def _create_time_series_section(self, df: pd.DataFrame, 
                                  analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Cria seção de análise de séries temporais"""
        return self._create_temporal_patterns_viz(df, analysis_results)
    
    def _create_filter_controls(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Cria controles de filtro interativo"""
        filters = {}
        
        # Filtros para colunas categóricas
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols[:5]:  # Limita a 5 colunas
            unique_values = df[col].dropna().unique()[:50]  # Máximo 50 valores
            filters[col] = {
                'type': 'select',
                'options': unique_values.tolist(),
                'default': 'all'
            }
        
        # Filtros para colunas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:5]:  # Limita a 5 colunas
            col_data = df[col].dropna()
            filters[col] = {
                'type': 'range',
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'default': [float(col_data.min()), float(col_data.max())]
            }
        
        return filters

# Instância global
dashboard_generator = DashboardGenerator()
