"""
Agente Autônomo para Análise de CSV - I2A2 Challenge
Especializado em detecção de fraudes e análise exploratória de dados
Autor: Seu Nome
Data: 28/09/2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
import json
import os
from datetime import datetime
import requests
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from scipy import stats
import io
import base64

warnings.filterwarnings('ignore')

@dataclass
class DataAnalysis:
    """Estrutura para armazenar resultados de análise"""
    summary_stats: Dict
    data_types: Dict
    missing_values: Dict
    correlations: pd.DataFrame
    outliers: Dict
    fraud_indicators: Optional[Dict] = None

class CSVMemory:
    """Sistema de memória para o agente"""
    def __init__(self):
        self.conversation_history = []
        self.analysis_cache = {}
        self.insights = []
        self.current_dataset_info = None
    
    def add_conversation(self, question: str, answer: str, analysis_type: str = "general"):
        """Adiciona interação à memória"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "analysis_type": analysis_type
        }
        self.conversation_history.append(entry)
    
    def add_insight(self, insight: str, category: str = "general"):
        """Adiciona insight descoberto"""
        self.insights.append({
            "timestamp": datetime.now().isoformat(),
            "insight": insight,
            "category": category
        })
    
    def get_context(self) -> str:
        """Retorna contexto relevante para o LLM"""
        context = f"Dataset atual: {self.current_dataset_info}\n"
        context += f"Insights descobertos: {len(self.insights)}\n"
        
        # Últimas 3 interações para contexto
        recent_conv = self.conversation_history[-3:] if self.conversation_history else []
        for conv in recent_conv:
            context += f"Q: {conv['question'][:100]}...\nA: {conv['answer'][:150]}...\n"
        
        return context

class OllamaReasoning:
    """Sistema de reasoning usando Ollama"""
    def __init__(self, model_name: str = "llama3.1"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434/api/generate"
    
    def query_ollama(self, prompt: str, system_prompt: str = None) -> str:
        """Consulta o modelo Ollama local"""
        try:
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_ctx": 4096
                }
            }
            
            response = requests.post(self.base_url, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json()["response"]
            else:
                return f"Erro na consulta ao Ollama: {response.status_code}"
                
        except Exception as e:
            return f"Erro ao conectar com Ollama: {str(e)}. Certifique-se que está rodando 'ollama serve'"
    
    def analyze_data_structure(self, df: pd.DataFrame, memory_context: str = "") -> str:
        """Analisa a estrutura dos dados usando reasoning"""
        
        system_prompt = """Você é um especialista em análise de dados e detecção de fraudes. 
        Analise a estrutura dos dados fornecidos e forneça insights sobre possíveis padrões, 
        anomalias e características relevantes para detecção de fraude. Seja conciso e técnico."""
        
        # Preparar informações sobre o dataset
        data_info = f"""
        DATASET INFORMATION:
        - Linhas: {df.shape[0]:,}
        - Colunas: {df.shape[1]}
        - Colunas: {list(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}
        - Dados faltantes: {df.isnull().sum().sum()}
        - Tipos: {df.dtypes.value_counts().to_dict()}
        
        ESTATÍSTICAS NUMÉRICAS:
        {df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else 'Sem dados numéricos'}
        
        CONTEXTO ANTERIOR:
        {memory_context[:500]}
        
        Analise e forneça insights sobre:
        1. Possíveis indicadores de fraude
        2. Qualidade dos dados
        3. Padrões interessantes
        4. Recomendações de análise
        """
        
        return self.query_ollama(data_info, system_prompt)
    
    def interpret_analysis_results(self, analysis: DataAnalysis, question: str, memory_context: str = "") -> str:
        """Interpreta resultados de análise usando reasoning"""
        
        system_prompt = """Você é um especialista em análise de dados focado em detecção de fraudes. 
        Interprete os resultados da análise e responda à pergunta do usuário de forma clara, 
        técnica e prática. Forneça insights actionáveis."""
        
        analysis_summary = f"""
        PERGUNTA: {question}
        
        RESULTADOS DA ANÁLISE:
        - Estatísticas: {str(analysis.summary_stats)[:500]}...
        - Valores ausentes: {analysis.missing_values}
        - Outliers: {str(analysis.outliers)[:300]}...
        - Fraude info: {analysis.fraud_indicators}
        
        CONTEXTO: {memory_context[:300]}
        
        Responda de forma clara e prática com insights específicos.
        """
        
        return self.query_ollama(analysis_summary, system_prompt)

class CSVAgent:
    """Agente principal para análise de CSV"""
    
    def __init__(self, ollama_model: str = "llama3.1"):
        self.memory = CSVMemory()
        self.reasoning = OllamaReasoning(ollama_model)
        self.current_df = None
        self.current_analysis = None
        
    def load_csv(self, file_path: str) -> bool:
        """Carrega arquivo CSV"""
        try:
            print(f"🔄 Carregando CSV: {file_path}")
            
            # Tentar diferentes encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    self.current_df = pd.read_csv(file_path, encoding=encoding)
                    print(f"✅ CSV carregado com encoding {encoding}")
                    print(f"📊 Shape: {self.current_df.shape}")
                    break
                except UnicodeDecodeError:
                    continue
            
            if self.current_df is None:
                print("❌ Erro: Não foi possível carregar o CSV")
                return False
            
            # Armazenar info na memória
            self.memory.current_dataset_info = {
                "filename": os.path.basename(file_path),
                "shape": self.current_df.shape,
                "columns": list(self.current_df.columns),
                "loaded_at": datetime.now().isoformat()
            }
            
            # Adicionar insight automático
            self.memory.add_insight(
                f"Dataset carregado: {self.current_df.shape[0]:,} linhas, {self.current_df.shape[1]} colunas",
                "data_loading"
            )
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar CSV: {str(e)}")
            return False
    
    def perform_eda(self) -> DataAnalysis:
        """Realiza análise exploratória completa"""
        if self.current_df is None:
            raise ValueError("Nenhum dataset carregado")
        
        print("🔍 Realizando análise exploratória...")
        df = self.current_df
        
        # Estatísticas básicas
        summary_stats = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            summary_stats[col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'q25': float(df[col].quantile(0.25)),
                'q75': float(df[col].quantile(0.75))
            }
        
        # Tipos de dados
        data_types = {str(k): str(v) for k, v in df.dtypes.to_dict().items()}
        
        # Valores ausentes
        missing_values = {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()}
        
        # Correlações (apenas primeiras 10 colunas numéricas para performance)
        correlations = None
        if len(numeric_cols) > 1:
            corr_cols = numeric_cols[:10]  # Limitar para performance
            correlations = df[corr_cols].corr()
        
        # Detecção de outliers usando IQR
        outliers = {}
        for col in numeric_cols[:5]:  # Limitar para as primeiras 5 colunas
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outliers[col] = {
                'count': int(outlier_mask.sum()),
                'percentage': float((outlier_mask.sum() / len(df)) * 100),
                'bounds': [float(lower_bound), float(upper_bound)]
            }
        
        # Análise específica de fraude se coluna 'Class' existir
        fraud_indicators = None
        if 'Class' in df.columns:
            fraud_count = int(df['Class'].sum())
            total_count = len(df)
            fraud_indicators = {
                'fraud_count': fraud_count,
                'fraud_percentage': float((fraud_count / total_count) * 100),
                'normal_count': total_count - fraud_count,
                'class_distribution': {str(k): int(v) for k, v in df['Class'].value_counts().to_dict().items()}
            }
            
            # Adicionar insight sobre fraude
            self.memory.add_insight(
                f"Taxa de fraude detectada: {fraud_indicators['fraud_percentage']:.2f}%",
                "fraud_detection"
            )
        
        self.current_analysis = DataAnalysis(
            summary_stats=summary_stats,
            data_types=data_types,
            missing_values=missing_values,
            correlations=correlations,
            outliers=outliers,
            fraud_indicators=fraud_indicators
        )
        
        print("✅ Análise exploratória concluída")
        return self.current_analysis
    
    def generate_visualization(self, viz_type: str, columns: List[str] = None) -> str:
        """Gera visualizações dos dados"""
        if self.current_df is None:
            return "Erro: Nenhum dataset carregado"
        
        df = self.current_df
        plt.style.use('default')
        
        try:
            if viz_type == "correlation_heatmap":
                numeric_df = df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) > 1:
                    # Limitar a 15 colunas para visualização
                    cols_to_plot = numeric_df.columns[:15]
                    
                    plt.figure(figsize=(12, 10))
                    correlation_matrix = numeric_df[cols_to_plot].corr()
                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                              fmt='.2f', square=True, cbar_kws={'label': 'Correlação'})
                    plt.title('Matriz de Correlação', fontsize=16, fontweight='bold')
                    plt.tight_layout()
                    
                    # Salvar como base64
                    img_buffer = io.BytesIO()
                    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                    img_buffer.seek(0)
                    img_str = base64.b64encode(img_buffer.read()).decode()
                    plt.close()
                    
                    return f"data:image/png;base64,{img_str}"
                else:
                    return "Erro: Não há colunas numéricas suficientes para correlação"
            
            elif viz_type == "fraud_analysis" and 'Class' in df.columns:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                
                # 1. Distribuição de classes
                class_counts = df['Class'].value_counts()
                axes[0,0].bar(['Normal', 'Fraude'], [class_counts[0], class_counts[1]], 
                             color=['skyblue', 'salmon'])
                axes[0,0].set_title('Distribuição: Normal vs Fraude', fontweight='bold')
                axes[0,0].set_ylabel('Quantidade')
                
                # Adicionar valores nas barras
                for i, v in enumerate([class_counts[0], class_counts[1]]):
                    axes[0,0].text(i, v + max(class_counts) * 0.01, f'{v:,}', 
                                  ha='center', va='bottom', fontweight='bold')
                
                # 2. Distribuição de Amount por classe (se existir)
                if 'Amount' in df.columns:
                    df.boxplot(column='Amount', by='Class', ax=axes[0,1])
                    axes[0,1].set_title('Distribuição de Valores por Classe')
                    axes[0,1].set_xlabel('Classe (0=Normal, 1=Fraude)')
                
                # 3. Distribuição temporal (se Time existir)
                if 'Time' in df.columns:
                    normal_time = df[df['Class']==0]['Time']
                    fraud_time = df[df['Class']==1]['Time']
                    
                    axes[1,0].hist([normal_time, fraud_time], bins=50, alpha=0.7, 
                                  label=['Normal', 'Fraude'], color=['skyblue', 'salmon'])
                    axes[1,0].set_title('Distribuição Temporal das Transações')
                    axes[1,0].set_xlabel('Time')
                    axes[1,0].legend()
                
                # 4. Top correlações com fraude
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                correlations_with_class = df[numeric_cols].corrwith(df['Class']).sort_values(key=abs, ascending=False)
                top_corr = correlations_with_class.head(10)
                
                colors = ['red' if x > 0 else 'blue' for x in top_corr.values]
                axes[1,1].barh(range(len(top_corr)), top_corr.values, color=colors, alpha=0.7)
                axes[1,1].set_yticks(range(len(top_corr)))
                axes[1,1].set_yticklabels(top_corr.index)
                axes[1,1].set_title('Top 10 Correlações com Fraude')
                axes[1,1].set_xlabel('Correlação')
                
                plt.tight_layout()
                
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                img_buffer.seek(0)
                img_str = base64.b64encode(img_buffer.read()).decode()
                plt.close()
                
                return f"data:image/png;base64,{img_str}"
            
            elif viz_type == "distribution" and columns:
                col = columns[0]
                if col in df.columns:
                    plt.figure(figsize=(10, 6))
                    
                    if df[col].dtype in ['int64', 'float64']:
                        # Histograma para dados numéricos
                        plt.hist(df[col].dropna(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                        plt.title(f'Distribuição de {col}', fontsize=14, fontweight='bold')
                        plt.xlabel(col)
                        plt.ylabel('Frequência')
                        
                        # Adicionar estatísticas
                        mean_val = df[col].mean()
                        median_val = df[col].median()
                        plt.axvline(mean_val, color='red', linestyle='--', label=f'Média: {mean_val:.2f}')
                        plt.axvline(median_val, color='green', linestyle='--', label=f'Mediana: {median_val:.2f}')
                        plt.legend()
                    else:
                        # Gráfico de barras para dados categóricos
                        value_counts = df[col].value_counts().head(20)
                        plt.bar(range(len(value_counts)), value_counts.values, color='skyblue')
                        plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)
                        plt.title(f'Distribuição de {col}', fontsize=14, fontweight='bold')
                        plt.ylabel('Frequência')
                    
                    plt.tight_layout()
                    
                    img_buffer = io.BytesIO()
                    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                    img_buffer.seek(0)
                    img_str = base64.b64encode(img_buffer.read()).decode()
                    plt.close()
                    
                    return f"data:image/png;base64,{img_str}"
                else:
                    return f"Erro: Coluna '{col}' não encontrada no dataset"
                    
        except Exception as e:
            plt.close('all')  # Limpar figuras em caso de erro
            return f"Erro ao gerar visualização: {str(e)}"
        
        return "Tipo de visualização não reconhecido"
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Responde pergunta sobre os dados"""
        if self.current_df is None:
            return {"error": "Nenhum dataset carregado"}
        
        print(f"🤔 Processando pergunta: {question[:50]}...")
        
        # Realizar EDA se ainda não foi feita
        if self.current_analysis is None:
            self.perform_eda()
        
        # Obter contexto da memória
        memory_context = self.memory.get_context()
        
        # Usar reasoning para interpretar a pergunta e gerar resposta
        llm_response = self.reasoning.interpret_analysis_results(
            self.current_analysis, question, memory_context
        )
        
        # Determinar se precisa de visualização
        viz_keywords = ["gráfico", "visualização", "chart", "plot", "distribuição", "correlação", "mostrar", "visual"]
        needs_viz = any(keyword in question.lower() for keyword in viz_keywords)
        
        response = {
            "answer": llm_response,
            "visualization": None,
            "data_insights": []
        }
        
        # Gerar visualização apropriada se necessário
        if needs_viz:
            if "correlação" in question.lower():
                print("📊 Gerando matriz de correlação...")
                viz_data = self.generate_visualization("correlation_heatmap")
                if viz_data.startswith("data:image"):
                    response["visualization"] = viz_data
            elif any(word in question.lower() for word in ["fraude", "fraud", "class"]) and 'Class' in self.current_df.columns:
                print("🚨 Gerando análise de fraude...")
                viz_data = self.generate_visualization("fraud_analysis")
                if viz_data.startswith("data:image"):
                    response["visualization"] = viz_data
        
        # Adicionar insights baseados na análise
        if self.current_analysis.fraud_indicators:
            fraud_rate = self.current_analysis.fraud_indicators['fraud_percentage']
            response["data_insights"].append(f"Taxa de fraude: {fraud_rate:.2f}%")
            
            if fraud_rate > 1:
                response["data_insights"].append("⚠️ Alta taxa de fraude detectada")
            elif fraud_rate < 0.1:
                response["data_insights"].append("✅ Baixa taxa de fraude - dataset balanceado")
        
        # Adicionar à memória
        self.memory.add_conversation(question, llm_response)
        
        print("✅ Resposta gerada com sucesso")
        return response
    
    def get_conclusions(self) -> str:
        """Gera conclusões gerais sobre o dataset"""
        if self.current_analysis is None:
            self.perform_eda()
        
        print("🧠 Gerando conclusões detalhadas...")
        memory_context = self.memory.get_context()
        
        system_prompt = """Você é um especialista em análise de dados e detecção de fraudes. 
        Com base em toda a análise realizada, forneça conclusões abrangentes e práticas 
        sobre o dataset. Seja específico e técnico, mas também prático."""
        
        # Preparar dados para análise de conclusões
        dataset_summary = ""
        if self.current_df is not None:
            dataset_summary = f"""
            RESUMO DO DATASET:
            - Total de registros: {self.current_df.shape[0]:,}
            - Colunas: {self.current_df.shape[1]}
            - Dados faltantes: {self.current_df.isnull().sum().sum()}
            - Tipos de dados: {self.current_df.dtypes.value_counts().to_dict()}
            """
        
        conclusions_prompt = f"""
        {dataset_summary}
        
        ANÁLISE REALIZADA:
        - Estatísticas calculadas: {len(self.current_analysis.summary_stats)} colunas numéricas
        - Outliers detectados: {len(self.current_analysis.outliers)} colunas analisadas
        - Indicadores de fraude: {self.current_analysis.fraud_indicators}
        
        HISTÓRICO DE INTERAÇÕES:
        {memory_context}
        
        Com base nesta análise completa, forneça conclusões sobre:
        1. Qualidade e características dos dados
        2. Padrões de fraude identificados (se aplicável)
        3. Principais insights descobertos
        4. Recomendações para ações futuras
        5. Limitações e próximos passos sugeridos
        
        Seja específico, técnico e prático nas recomendações.
        """
        
        conclusions = self.reasoning.query_ollama(conclusions_prompt, system_prompt)
        
        # Adicionar à memória
        self.memory.add_conversation("Conclusões gerais do dataset", conclusions, "conclusions")
        
        print("✅ Conclusões geradas")
        return conclusions

# Função de teste
def test_agent():
    """Função para testar o agente"""
    print("🤖 Testando Agente CSV I2A2")
    print("=" * 50)
    
    agent = CSVAgent()
    
    # Testar carregamento (usar um arquivo pequeno para teste)
    test_file = "../creditcard.csv"  # Ajustar caminho se necessário
    
    if os.path.exists(test_file):
        print(f"📁 Testando com arquivo: {test_file}")
        success = agent.load_csv(test_file)
        
        if success:
            print("✅ CSV carregado com sucesso")
            
            # Teste básico de pergunta
            response = agent.answer_question("Quantas linhas tem o dataset?")
            print(f"🤖 Resposta: {response['answer'][:200]}...")
            
        else:
            print("❌ Erro ao carregar CSV")
    else:
        print(f"❌ Arquivo não encontrado: {test_file}")
    
    return agent

if __name__ == "__main__":
    test_agent()
