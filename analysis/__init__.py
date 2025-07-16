"""
Módulo de análise do Intelligent CSV Processor
Contém analisadores estatísticos, detectores de padrões e análise de correlações
"""

from .statistical_analyzer import StatisticalAnalyzer, statistical_analyzer
from .pattern_detector import PatternDetector, pattern_detector
from .correlation_finder import CorrelationFinder, correlation_finder

__all__ = [
    'StatisticalAnalyzer', 'statistical_analyzer',
    'PatternDetector', 'pattern_detector', 
    'CorrelationFinder', 'correlation_finder'
]
