import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ChartBuilder:
    """Construtor especializado de gráficos baseado no tipo de dados"""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
        self.template = "plotly_white"
        
    def build_chart(self, data: pd.DataFrame, chart_type: str, 
                   column: str = None, **kwargs) -> Dict[str, Any]:
        """Factory method para construir gráficos baseado no tipo"""
        
        chart_builders = {
            'histogram': self._build_histogram,
            'boxplot': self._build_boxplot,
            'scatter': self._build_scatter,
            'line': self._build_line_chart,
            'bar': self._build_bar_chart,
            'pie': self._build_pie_chart,
            'heatmap': self._build_heatmap,
            'violin': self._build_violin_plot,
            'distribution': self._build_distribution_plot,
            'correlation': self._build_correlation_plot,
            'time_series': self._build_time_series,
            'categorical': self._build_categorical_chart
        }
        
        if chart_type not in chart_builders:
            raise ValueError(f"Tipo de gráfico não suportado: {chart_type}")
        
        try:
            return chart_builders[chart_type](data, column, **kwargs)
        except Exception as e:
            logger.error(f"Erro ao construir gráfico {chart_type}: {e}")
            return {'error': str(e), 'chart_type': chart_type}
    
    def _build_histogram(self, data: pd.DataFrame, column: str, **kwargs) -> Dict[str, Any]:
        """Constrói histograma para dados numéricos"""
        if column not in data.columns:
            return {'error': f'Coluna {column} não encontrada'}
        
        series = data[column].dropna()
        if len(series) == 0:
            return {'error': 'Dados insuficientes para histograma'}
        
        bins = kwargs.get('bins', 30)
        title = kwargs.get('title', f'Distribuição de {column}')
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=series,
            nbinsx=bins,
            name=column,
            marker_color=self.color_palette[0],
            opacity=0.7
        ))
        
        # Adiciona linha de média
        mean_val = series.mean()
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Média: {mean_val:.2f}"
        )
        
        # Adiciona linha de mediana
        median_val = series.median()
        fig.add_vline(
            x=median_val,
            line_dash="dot",
            line_color="blue",
            annotation_text=f"Mediana: {median_val:.2f}"
        )
        
        fig.update_layout(
            title=title,
            xaxis_title=column,
            yaxis_title="Frequência",
            template=self.template,
            showlegend=False
        )
        
        return {
            'chart': fig.to_json(),
            'chart_type': 'histogram',
            'stats': {
                'mean': float(mean_val),
                'median': float(median_val),
                'std': float(series.std()),
                'count': len(series)
            }
        }
    
    def _build_boxplot(self, data: pd.DataFrame, column: str, **kwargs) -> Dict[str, Any]:
        """Constrói boxplot para análise de outliers"""
        if column not in data.columns:
            return {'error': f'Coluna {column} não encontrada'}
        
        series = data[column].dropna()
        if len(series) == 0:
            return {'error': 'Dados insuficientes para boxplot'}
        
        title = kwargs.get('title', f'Boxplot de {column}')
        group_by = kwargs.get('group_by')
        
        fig = go.Figure()
        
        if group_by and group_by in data.columns:
            # Boxplot agrupado
            groups = data[group_by].unique()
            for i, group in enumerate(groups):
                group_data = data[data[group_by] == group][column].dropna()
                
                fig.add_trace(go.Box(
                    y=group_data,
                    name=str(group),
                    boxpoints='outliers',
                    marker_color=self.color_palette[i % len(self.color_palette)]
                ))
        else:
            # Boxplot simples
            fig.add_trace(go.Box(
                y=series,
                name=column,
                boxpoints='outliers',
                marker_color=self.color_palette[0]
            ))
        
        fig.update_layout(
            title=title,
            yaxis_title=column,
            template=self.template
        )
        
        return {
            'chart': fig.to_json(),
            'chart_type': 'boxplot'
        }
    
    def _build_scatter(self, data: pd.DataFrame, column: str, **kwargs) -> Dict[str, Any]:
        """Constrói gráfico de dispersão"""
        x_col = kwargs.get('x_column', column)
        y_col = kwargs.get('y_column')
        
        if not y_col:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            available_cols = [col for col in numeric_cols if col != x_col]
            if not available_cols:
                return {'error': 'Necessário especificar y_column para scatter plot'}
            y_col = available_cols[0]
        
        if x_col not in data.columns or y_col not in data.columns:
            return {'error': 'Colunas especificadas não encontradas'}
        
        clean_data = data[[x_col, y_col]].dropna()
        if len(clean_data) == 0:
            return {'error': 'Dados insuficientes para scatter plot'}
        
        title = kwargs.get('title', f'{y_col} vs {x_col}')
        color_by = kwargs.get('color_by')
        
        fig = go.Figure()
        
        if color_by and color_by in data.columns:
            # Scatter colorido por categoria
            categories = data[color_by].unique()
            for i, category in enumerate(categories):
                cat_data = clean_data[data[color_by] == category]
                
                fig.add_trace(go.Scatter(
                    x=cat_data[x_col],
                    y=cat_data[y_col],
                    mode='markers',
                    name=str(category),
                    marker=dict(
                        color=self.color_palette[i % len(self.color_palette)],
                        size=8,
                        opacity=0.7
                    )
                ))
        else:
            # Scatter simples
            fig.add_trace(go.Scatter(
                x=clean_data[x_col],
                y=clean_data[y_col],
                mode='markers',
                marker=dict(
                    color=self.color_palette[0],
                    size=8,
                    opacity=0.7
                ),
                name='Dados'
            ))
        
        # Adiciona linha de tendência
        if kwargs.get('trendline', True):
            z = np.polyfit(clean_data[x_col], clean_data[y_col], 1)
            p = np.poly1d(z)
            
            fig.add_trace(go.Scatter(
                x=clean_data[x_col],
                y=p(clean_data[x_col]),
                mode='lines',
                name='Tendência',
                line=dict(color='red', dash='dash')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title=y_col,
            template=self.template
        )
        
        return {
            'chart': fig.to_json(),
            'chart_type': 'scatter'
        }
    
    def _build_line_chart(self, data: pd.DataFrame, column: str, **kwargs) -> Dict[str, Any]:
        """Constrói gráfico de linhas"""
        if column not in data.columns:
            return {'error': f'Coluna {column} não encontrada'}
        
        x_col = kwargs.get('x_column')
        if not x_col:
            # Usa índice como x
            x_data = data.index
            x_title = 'Índice'
        else:
            if x_col not in data.columns:
                return {'error': f'Coluna X {x_col} não encontrada'}
            x_data = data[x_col]
            x_title = x_col
        
        y_data = data[column].dropna()
        title = kwargs.get('title', f'Série Temporal - {column}')
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines+markers',
            name=column,
            line=dict(color=self.color_palette[0])
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_title,
            yaxis_title=column,
            template=self.template
        )
        
        return {
            'chart': fig.to_json(),
            'chart_type': 'line'
        }
    
    def _build_bar_chart(self, data: pd.DataFrame, column: str, **kwargs) -> Dict[str, Any]:
        """Constrói gráfico de barras"""
        if column not in data.columns:
            return {'error': f'Coluna {column} não encontrada'}
        
        # Conta valores ou usa valores fornecidos
        value_col = kwargs.get('value_column')
        
        if value_col and value_col in data.columns:
            # Agrupa por categoria e soma valores
            chart_data = data.groupby(column)[value_col].sum().reset_index()
            y_values = chart_data[value_col]
            y_title = value_col
        else:
            # Conta frequências
            chart_data = data[column].value_counts().reset_index()
            chart_data.columns = [column, 'count']
            y_values = chart_data['count']
            y_title = 'Frequência'
        
        x_values = chart_data[column]
        title = kwargs.get('title', f'Distribuição de {column}')
        
        # Limita a 20 categorias para legibilidade
        if len(x_values) > 20:
            top_data = chart_data.nlargest(20, y_values.name)
            x_values = top_data[column]
            y_values = top_data[y_values.name]
            title += ' (Top 20)'
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=x_values,
            y=y_values,
            marker_color=self.color_palette[0],
            text=y_values,
            textposition='outside'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=column,
            yaxis_title=y_title,
            template=self.template,
            xaxis_tickangle=-45
        )
        
        return {
            'chart': fig.to_json(),
            'chart_type': 'bar'
        }
    
    def _build_pie_chart(self, data: pd.DataFrame, column: str, **kwargs) -> Dict[str, Any]:
        """Constrói gráfico de pizza"""
        if column not in data.columns:
            return {'error': f'Coluna {column} não encontrada'}
        
        value_counts = data[column].value_counts()
        
        # Limita a 10 categorias
        if len(value_counts) > 10:
            top_values = value_counts.head(9)
            others_sum = value_counts.tail(len(value_counts) - 9).sum()
            top_values['Outros'] = others_sum
            value_counts = top_values
        
        title = kwargs.get('title', f'Distribuição de {column}')
        
        fig = go.Figure()
        
        fig.add_trace(go.Pie(
            labels=value_counts.index,
            values=value_counts.values,
            hole=0.3,
            marker_colors=self.color_palette[:len(value_counts)]
        ))
        
        fig.update_layout(
            title=title,
            template=self.template
        )
        
        return {
            'chart': fig.to_json(),
            'chart_type': 'pie'
        }
    
    def _build_heatmap(self, data: pd.DataFrame, column: str = None, **kwargs) -> Dict[str, Any]:
        """Constrói mapa de calor"""
        matrix_data = kwargs.get('matrix_data')
        
        if matrix_data is not None:
            # Usa matriz fornecida
            heatmap_data = matrix_data
        else:
            # Calcula matriz de correlação
            numeric_data = data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) < 2:
                return {'error': 'Dados insuficientes para heatmap'}
            heatmap_data = numeric_data.corr()
        
        title = kwargs.get('title', 'Mapa de Calor')
        
        fig = go.Figure()
        
        fig.add_trace(go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.columns,
            colorscale='RdBu',
            zmid=0,
            text=heatmap_data.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title=title,
            template=self.template
        )
        
        return {
            'chart': fig.to_json(),
            'chart_type': 'heatmap'
        }
    
    def _build_violin_plot(self, data: pd.DataFrame, column: str, **kwargs) -> Dict[str, Any]:
        """Constrói violin plot"""
        if column not in data.columns:
            return {'error': f'Coluna {column} não encontrada'}
        
        series = data[column].dropna()
        if len(series) == 0:
            return {'error': 'Dados insuficientes para violin plot'}
        
        title = kwargs.get('title', f'Violin Plot - {column}')
        group_by = kwargs.get('group_by')
        
        fig = go.Figure()
        
        if group_by and group_by in data.columns:
            groups = data[group_by].unique()
            for i, group in enumerate(groups):
                group_data = data[data[group_by] == group][column].dropna()
                
                fig.add_trace(go.Violin(
                    y=group_data,
                    name=str(group),
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor=self.color_palette[i % len(self.color_palette)],
                    opacity=0.6
                ))
        else:
            fig.add_trace(go.Violin(
                y=series,
                name=column,
                box_visible=True,
                meanline_visible=True,
                fillcolor=self.color_palette[0],
                opacity=0.6
            ))
        
        fig.update_layout(
            title=title,
            yaxis_title=column,
            template=self.template
        )
        
        return {
            'chart': fig.to_json(),
            'chart_type': 'violin'
        }
    
    def _build_distribution_plot(self, data: pd.DataFrame, column: str, **kwargs) -> Dict[str, Any]:
        """Constrói gráfico de distribuição combinado"""
        if column not in data.columns:
            return {'error': f'Coluna {column} não encontrada'}
        
        series = data[column].dropna()
        if len(series) == 0:
            return {'error': 'Dados insuficientes para plot de distribuição'}
        
        title = kwargs.get('title', f'Análise de Distribuição - {column}')
        
        # Cria subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Histograma', 'Box Plot', 'Violin Plot', 'Q-Q Plot'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Histograma
        fig.add_trace(
            go.Histogram(x=series, nbinsx=30, name='Histograma'),
            row=1, col=1
        )
        
        # Box Plot
        fig.add_trace(
            go.Box(y=series, name='Box Plot'),
            row=1, col=2
        )
        
        # Violin Plot
        fig.add_trace(
            go.Violin(y=series, name='Violin Plot'),
            row=2, col=1
        )
        
        # Q-Q Plot (aproximado)
        from scipy import stats
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(series)))
        sample_quantiles = np.sort(series)
        
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sample_quantiles, 
                      mode='markers', name='Q-Q Plot'),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            template=self.template,
            showlegend=False
        )
        
        return {
            'chart': fig.to_json(),
            'chart_type': 'distribution'
        }
    
    def _build_correlation_plot(self, data: pd.DataFrame, column: str = None, **kwargs) -> Dict[str, Any]:
        """Constrói gráfico de correlação"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            return {'error': 'Necessário pelo menos 2 colunas numéricas'}
        
        # Limita colunas para performance
        if len(numeric_data.columns) > 15:
            numeric_data = numeric_data.iloc[:, :15]
        
        corr_matrix = numeric_data.corr()
        
        # Cria heatmap de correlação
        return self._build_heatmap(data, matrix_data=corr_matrix, 
                                 title='Matriz de Correlação')
    
    def _build_time_series(self, data: pd.DataFrame, column: str, **kwargs) -> Dict[str, Any]:
        """Constrói gráfico de série temporal"""
        date_col = kwargs.get('date_column')
        
        if not date_col:
            # Procura coluna de data automaticamente
            date_cols = data.select_dtypes(include=['datetime64']).columns
            if len(date_cols) == 0:
                return {'error': 'Nenhuma coluna de data encontrada'}
            date_col = date_cols[0]
        
        if date_col not in data.columns or column not in data.columns:
            return {'error': 'Colunas especificadas não encontradas'}
        
        # Ordena por data
        time_data = data[[date_col, column]].dropna().sort_values(date_col)
        
        if len(time_data) == 0:
            return {'error': 'Dados insuficientes para série temporal'}
        
        title = kwargs.get('title', f'Série Temporal - {column}')
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time_data[date_col],
            y=time_data[column],
            mode='lines+markers',
            name=column,
            line=dict(color=self.color_palette[0])
        ))
        
        # Adiciona média móvel se solicitado
        if kwargs.get('moving_average'):
            window = kwargs.get('ma_window', 7)
            ma = time_data[column].rolling(window=window).mean()
            
            fig.add_trace(go.Scatter(
                x=time_data[date_col],
                y=ma,
                mode='lines',
                name=f'Média Móvel ({window})',
                line=dict(color='red', dash='dash')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title=date_col,
            yaxis_title=column,
            template=self.template
        )
        
        return {
            'chart': fig.to_json(),
            'chart_type': 'time_series'
        }
    
    def _build_categorical_chart(self, data: pd.DataFrame, column: str, **kwargs) -> Dict[str, Any]:
        """Constrói gráfico adequado para dados categóricos"""
        if column not in data.columns:
            return {'error': f'Coluna {column} não encontrada'}
        
        unique_values = data[column].nunique()
        
        # Escolhe tipo de gráfico baseado no número de categorias
        if unique_values <= 10:
            return self._build_pie_chart(data, column, **kwargs)
        else:
            return self._build_bar_chart(data, column, **kwargs)

# Instância global
chart_builder = ChartBuilder()
