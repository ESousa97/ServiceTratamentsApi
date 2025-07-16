import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import logging
import re
from models.embeddings import embedding_engine

logger = logging.getLogger(__name__)

class TextClassifier:
    """Classificador de texto usando redes neurais"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        
    def classify_text_categories(self, texts: List[str]) -> Dict[str, Any]:
        """Classifica textos em categorias automáticas"""
        try:
            if not texts:
                return {'categories': [], 'confidence': 0}
            
            # Gera embeddings
            embeddings = embedding_engine.generate_embeddings_batch(texts)
            embeddings_array = np.array(embeddings)
            
            # Clustering para identificar categorias
            n_clusters = min(10, max(2, len(texts) // 5))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings_array)
            
            # Analisa categorias encontradas
            categories = {}
            for i, label in enumerate(cluster_labels):
                if label not in categories:
                    categories[label] = []
                categories[label].append(texts[i])
            
            # Gera nomes para categorias baseado no conteúdo
            category_names = {}
            for label, category_texts in categories.items():
                name = self._generate_category_name(category_texts)
                category_names[name] = {
                    'texts': category_texts[:5],  # Máximo 5 exemplos
                    'count': len(category_texts),
                    'percentage': (len(category_texts) / len(texts)) * 100
                }
            
            return {
                'categories': category_names,
                'total_categories': len(categories),
                'silhouette_score': self._calculate_silhouette_score(embeddings_array, cluster_labels)
            }
            
        except Exception as e:
            logger.error(f"Erro na classificação de categorias: {e}")
            return {'categories': {}, 'error': str(e)}
    
    def analyze_sentiment(self, texts: List[str]) -> Dict[str, Any]:
        """Análise de sentimentos em textos"""
        try:
            if not texts:
                return {'sentiments': [], 'summary': {}}
            
            # Análise simplificada baseada em palavras-chave
            positive_words = ['bom', 'ótimo', 'excelente', 'positivo', 'sucesso', 'satisfeito']
            negative_words = ['ruim', 'péssimo', 'terrível', 'negativo', 'problema', 'insatisfeito']
            
            sentiments = []
            for text in texts:
                text_lower = text.lower()
                
                positive_count = sum(1 for word in positive_words if word in text_lower)
                negative_count = sum(1 for word in negative_words if word in text_lower)
                
                if positive_count > negative_count:
                    sentiment = 'positive'
                    score = min(1.0, positive_count / (positive_count + negative_count + 1))
                elif negative_count > positive_count:
                    sentiment = 'negative'
                    score = min(1.0, negative_count / (positive_count + negative_count + 1))
                else:
                    sentiment = 'neutral'
                    score = 0.5
                
                sentiments.append({
                    'text': text[:50] + '...' if len(text) > 50 else text,
                    'sentiment': sentiment,
                    'score': score
                })
            
            # Resumo
            sentiment_counts = {}
            for item in sentiments:
                sentiment = item['sentiment']
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            
            summary = {
                'distribution': sentiment_counts,
                'dominant_sentiment': max(sentiment_counts, key=sentiment_counts.get),
                'average_score': np.mean([item['score'] for item in sentiments])
            }
            
            return {
                'sentiments': sentiments[:10],  # Máximo 10 exemplos
                'summary': summary,
                'total_analyzed': len(texts)
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de sentimentos: {e}")
            return {'sentiments': [], 'error': str(e)}
    
    def extract_entities(self, texts: List[str]) -> Dict[str, Any]:
        """Extração de entidades nomeadas"""
        try:
            if not texts:
                return {'entities': {}, 'summary': {}}
            
            # Padrões regex para entidades comuns
            patterns = {
                'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'phone': r'\b(?:\+?55\s?)?(?:\(\d{2}\)|\d{2})\s?\d{4,5}-?\d{4}\b',
                'cpf': r'\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b',
                'cnpj': r'\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b',
                'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                'currency': r'R\$\s?\d{1,3}(?:\.\d{3})*(?:,\d{2})?',
                'url': r'https?://[^\s]+',
                'cep': r'\b\d{5}-?\d{3}\b'
            }
            
            entities = {}
            for entity_type, pattern in patterns.items():
                entities[entity_type] = []
                
                for text in texts:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    entities[entity_type].extend(matches)
                
                # Remove duplicatas
                entities[entity_type] = list(set(entities[entity_type]))
            
            # Resumo
            summary = {
                'total_entities': sum(len(ents) for ents in entities.values()),
                'entity_types_found': len([k for k, v in entities.items() if v]),
                'most_common_type': max(entities, key=lambda k: len(entities[k])) if entities else None
            }
            
            return {
                'entities': {k: v[:10] for k, v in entities.items() if v},  # Máximo 10 por tipo
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"Erro na extração de entidades: {e}")
            return {'entities': {}, 'error': str(e)}
    
    def detect_numeric_patterns(self, values: np.ndarray) -> Dict[str, Any]:
        """Detecta padrões em sequências numéricas"""
        try:
            if len(values) < 3:
                return {'patterns': [], 'message': 'Dados insuficientes'}
            
            patterns = []
            
            # Padrão aritmético
            diffs = np.diff(values)
            if np.std(diffs) < 1e-6:  # Diferença constante
                patterns.append({
                    'type': 'arithmetic',
                    'description': f'Progressão aritmética com diferença {diffs[0]:.2f}',
                    'confidence': 0.95
                })
            
            # Padrão geométrico
            if not np.any(values == 0):
                ratios = values[1:] / values[:-1]
                if np.std(ratios) < 1e-6:  # Razão constante
                    patterns.append({
                        'type': 'geometric',
                        'description': f'Progressão geométrica com razão {ratios[0]:.2f}',
                        'confidence': 0.95
                    })
            
            # Padrão cíclico (usando FFT)
            if len(values) > 10:
                fft = np.fft.fft(values - np.mean(values))
                frequencies = np.fft.fftfreq(len(values))
                dominant_freq = frequencies[np.argmax(np.abs(fft[1:len(fft)//2])) + 1]
                
                if abs(dominant_freq) > 0.01:  # Threshold para ciclicidade
                    period = int(1 / abs(dominant_freq))
                    patterns.append({
                        'type': 'cyclic',
                        'description': f'Padrão cíclico com período aproximado de {period}',
                        'confidence': 0.7
                    })
            
            # Padrão de crescimento/decrescimento
            if len(values) > 5:
                trend_correlation = np.corrcoef(np.arange(len(values)), values)[0, 1]
                if abs(trend_correlation) > 0.7:
                    trend_type = 'crescente' if trend_correlation > 0 else 'decrescente'
                    patterns.append({
                        'type': 'trend',
                        'description': f'Tendência {trend_type} (correlação: {trend_correlation:.2f})',
                        'confidence': min(0.9, abs(trend_correlation))
                    })
            
            return {
                'patterns': patterns,
                'total_patterns': len(patterns)
            }
            
        except Exception as e:
            logger.error(f"Erro na detecção de padrões numéricos: {e}")
            return {'patterns': [], 'error': str(e)}
    
    def detect_text_patterns(self, texts: List[str]) -> Dict[str, Any]:
        """Detecta padrões em sequências de texto"""
        try:
            if len(texts) < 3:
                return {'patterns': [], 'message': 'Dados insuficientes'}
            
            patterns = []
            
            # Padrão de comprimento
            lengths = [len(text) for text in texts]
            if np.std(lengths) < np.mean(lengths) * 0.1:  # Baixa variabilidade
                patterns.append({
                    'type': 'length_consistency',
                    'description': f'Textos com comprimento consistente (~{np.mean(lengths):.0f} caracteres)',
                    'confidence': 0.8
                })
            
            # Padrão de formato
            format_patterns = self._detect_format_patterns(texts)
            patterns.extend(format_patterns)
            
            # Padrão de vocabulário
            vocab_pattern = self._detect_vocabulary_pattern(texts)
            if vocab_pattern:
                patterns.append(vocab_pattern)
            
            # Padrão sequencial
            sequential_pattern = self._detect_sequential_pattern(texts)
            if sequential_pattern:
                patterns.append(sequential_pattern)
            
            return {
                'patterns': patterns,
                'total_patterns': len(patterns)
            }
            
        except Exception as e:
            logger.error(f"Erro na detecção de padrões textuais: {e}")
            return {'patterns': [], 'error': str(e)}
    
    def _generate_category_name(self, texts: List[str]) -> str:
        """Gera nome para categoria baseado no conteúdo"""
        try:
            # Extrai palavras mais frequentes
            all_words = []
            for text in texts[:10]:  # Máximo 10 textos para análise
                words = re.findall(r'\b\w+\b', text.lower())
                all_words.extend([w for w in words if len(w) > 3])
            
            if not all_words:
                return f"Categoria_{len(texts)}_itens"
            
            # Conta frequências
            word_counts = {}
            for word in all_words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # Pega palavra mais frequente
            most_common_word = max(word_counts, key=word_counts.get)
            
            return f"Categoria_{most_common_word}"
            
        except:
            return f"Categoria_{len(texts)}_itens"
    
    def _calculate_silhouette_score(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Calcula silhouette score para avaliar qualidade do clustering"""
        try:
            from sklearn.metrics import silhouette_score
            if len(set(labels)) > 1 and len(embeddings) > 1:
                return float(silhouette_score(embeddings, labels))
            return 0.0
        except:
            return 0.0
    
    def _detect_format_patterns(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Detecta padrões de formato em textos"""
        patterns = []
        
        # Padrões comuns
        format_checks = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,},
            'phone': r'^\+?[\d\s\-\(\)]{10,},
            'cpf': r'^\d{3}\.\d{3}\.\d{3}-\d{2},
            'date': r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{4},
            'number': r'^\d+,
            'alphanumeric': r'^[a-zA-Z0-9]+
        }
        
        for pattern_name, regex in format_checks.items():
            matches = sum(1 for text in texts if re.match(regex, text.strip()))
            match_percentage = (matches / len(texts)) * 100
            
            if match_percentage > 70:  # Threshold para considerar padrão
                patterns.append({
                    'type': f'format_{pattern_name}',
                    'description': f'{match_percentage:.0f}% dos textos seguem formato {pattern_name}',
                    'confidence': min(0.95, match_percentage / 100)
                })
        
        return patterns
    
    def _detect_vocabulary_pattern(self, texts: List[str]) -> Optional[Dict[str, Any]]:
        """Detecta padrão de vocabulário compartilhado"""
        try:
            # Extrai vocabulário único de cada texto
            vocabularies = []
            for text in texts[:20]:  # Limita para performance
                words = set(re.findall(r'\b\w+\b', text.lower()))
                vocabularies.append(words)
            
            if not vocabularies:
                return None
            
            # Calcula sobreposição de vocabulário
            common_words = vocabularies[0]
            for vocab in vocabularies[1:]:
                common_words = common_words.intersection(vocab)
            
            if len(common_words) > 2:  # Pelo menos 3 palavras em comum
                avg_vocab_size = np.mean([len(vocab) for vocab in vocabularies])
                overlap_percentage = (len(common_words) / avg_vocab_size) * 100
                
                return {
                    'type': 'vocabulary_overlap',
                    'description': f'Vocabulário compartilhado de {len(common_words)} palavras ({overlap_percentage:.0f}%)',
                    'confidence': min(0.9, overlap_percentage / 50)
                }
            
            return None
            
        except:
            return None
    
    def _detect_sequential_pattern(self, texts: List[str]) -> Optional[Dict[str, Any]]:
        """Detecta padrões sequenciais em textos"""
        try:
            # Verifica se há numeração sequencial
            numbers = []
            for text in texts:
                # Procura por números no início do texto
                match = re.match(r'^(\d+)', text.strip())
                if match:
                    numbers.append(int(match.group(1)))
            
            if len(numbers) > len(texts) * 0.7:  # 70% dos textos têm numeração
                # Verifica se é sequencial
                sorted_numbers = sorted(numbers)
                is_sequential = all(
                    sorted_numbers[i] == sorted_numbers[i-1] + 1 
                    for i in range(1, len(sorted_numbers))
                )
                
                if is_sequential:
                    return {
                        'type': 'sequential_numbering',
                        'description': f'Numeração sequencial detectada ({min(numbers)}-{max(numbers)})',
                        'confidence': 0.9
                    }
            
            return None
            
        except:
            return None

class AnomalyDetector:
    """Detector de anomalias usando redes neurais"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        
    def detect_anomalies(self, data: np.ndarray) -> Dict[str, Any]:
        """Detecta anomalias em dados numéricos"""
        try:
            if len(data) < 5:
                return {'anomaly_indices': [], 'anomaly_scores': [], 'threshold': 0}
            
            # Normaliza dados
            data_scaled = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
            
            # Usa Isolation Forest simplificado
            from sklearn.ensemble import IsolationForest
            
            detector = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = detector.fit_predict(data_scaled.reshape(-1, 1))
            
            # Extrai índices de anomalias
            anomaly_indices = np.where(anomaly_labels == -1)[0].tolist()
            
            # Calcula scores de anomalia
            anomaly_scores = detector.decision_function(data_scaled.reshape(-1, 1))
            threshold = np.percentile(anomaly_scores, 10)  # 10% como threshold
            
            return {
                'anomaly_indices': anomaly_indices,
                'anomaly_scores': anomaly_scores.tolist(),
                'threshold': float(threshold),
                'total_anomalies': len(anomaly_indices)
            }
            
        except Exception as e:
            logger.error(f"Erro na detecção de anomalias: {e}")
            return {'anomaly_indices': [], 'anomaly_scores': [], 'threshold': 0, 'error': str(e)}
    
    def detect_text_anomalies(self, texts: List[str]) -> Dict[str, Any]:
        """Detecta anomalias em dados textuais"""
        try:
            if len(texts) < 5:
                return {'anomaly_indices': [], 'anomaly_texts': [], 'reasons': []}
            
            anomalies = []
            
            # Anomalias por comprimento
            lengths = [len(text) for text in texts]
            mean_length = np.mean(lengths)
            std_length = np.std(lengths)
            
            for i, text in enumerate(texts):
                reasons = []
                
                # Comprimento anômalo
                if abs(len(text) - mean_length) > 3 * std_length:
                    reasons.append('comprimento_anomalo')
                
                # Caracteres especiais excessivos
                special_char_ratio = len(re.findall(r'[^\w\s]', text)) / len(text) if len(text) > 0 else 0
                if special_char_ratio > 0.5:
                    reasons.append('caracteres_especiais_excessivos')
                
                # Texto muito repetitivo
                words = text.split()
                if len(words) > 3:
                    unique_words = len(set(words))
                    repetition_ratio = unique_words / len(words)
                    if repetition_ratio < 0.3:
                        reasons.append('texto_repetitivo')
                
                # Texto vazio ou muito curto
                if len(text.strip()) < 3:
                    reasons.append('texto_muito_curto')
                
                if reasons:
                    anomalies.append({
                        'index': i,
                        'text': text[:100] + '...' if len(text) > 100 else text,
                        'reasons': reasons
                    })
            
            return {
                'anomaly_indices': [a['index'] for a in anomalies],
                'anomaly_details': anomalies[:10],  # Máximo 10 exemplos
                'total_text_anomalies': len(anomalies)
            }
            
        except Exception as e:
            logger.error(f"Erro na detecção de anomalias textuais: {e}")
            return {'anomaly_indices': [], 'anomaly_details': [], 'error': str(e)}

class AutoEncoder(nn.Module):
    """Autoencoder para detecção de anomalias"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = None):
        super(AutoEncoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [input_dim // 2, input_dim // 4, input_dim // 8]
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        hidden_dims_reversed = hidden_dims[::-1][1:] + [input_dim]
        for hidden_dim in hidden_dims_reversed:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU() if hidden_dim != input_dim else nn.Identity()
            ])
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

class NeuralAnomalyDetector:
    """Detector de anomalias usando autoencoder"""
    
    def __init__(self, input_dim: int):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoEncoder(input_dim).to(self.device)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def train(self, data: np.ndarray, epochs: int = 100, lr: float = 0.001):
        """Treina o autoencoder"""
        try:
            # Normaliza dados
            data_scaled = self.scaler.fit_transform(data)
            data_tensor = torch.FloatTensor(data_scaled).to(self.device)
            
            # Configuração do treinamento
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            
            self.model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                
                reconstructed, _ = self.model(data_tensor)
                loss = criterion(reconstructed, data_tensor)
                
                loss.backward()
                optimizer.step()
                
                if epoch % 20 == 0:
                    logger.debug(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
            self.is_trained = True
            logger.info("Autoencoder treinado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro no treinamento do autoencoder: {e}")
            
    def detect_anomalies_neural(self, data: np.ndarray, threshold_percentile: float = 95) -> Dict[str, Any]:
        """Detecta anomalias usando o autoencoder treinado"""
        try:
            if not self.is_trained:
                self.train(data)
            
            # Normaliza dados
            data_scaled = self.scaler.transform(data)
            data_tensor = torch.FloatTensor(data_scaled).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                reconstructed, _ = self.model(data_tensor)
                
            # Calcula erro de reconstrução
            reconstruction_errors = torch.mean((data_tensor - reconstructed) ** 2, dim=1)
            reconstruction_errors = reconstruction_errors.cpu().numpy()
            
            # Define threshold
            threshold = np.percentile(reconstruction_errors, threshold_percentile)
            
            # Identifica anomalias
            anomaly_mask = reconstruction_errors > threshold
            anomaly_indices = np.where(anomaly_mask)[0].tolist()
            
            return {
                'anomaly_indices': anomaly_indices,
                'reconstruction_errors': reconstruction_errors.tolist(),
                'threshold': float(threshold),
                'total_anomalies': len(anomaly_indices)
            }
            
        except Exception as e:
            logger.error(f"Erro na detecção neural de anomalias: {e}")
            return {'anomaly_indices': [], 'reconstruction_errors': [], 'threshold': 0, 'error': str(e)}
        