"""
Versão simplificada do agente CSV para teste (sem Ollama)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import io
import base64

warnings.filterwarnings('ignore')

class CSVAgentSimple:
    """Versão simplificada do agente para teste"""
    
    def __init__(self):
        self.current_df = None
        self.conversation_history = []
        
    def load_csv(self, file_path: str) -> bool:
        """Carrega arquivo CSV"""
        try:
            print(f"🔄 Carregando CSV: {file_path}")
            self.current_df = pd.read_csv(file_path)
            print(f"✅ CSV carregado - Shape: {self.current_df.shape}")
            return True
        except Exception as e:
            print(f"❌ Erro: {str(e)}")
            return False
    
    def get_basic_info(self) -> Dict:
        """Retorna informações básicas do dataset"""
        if self.current_df is None:
            return {"error": "Nenhum dataset carregado"}
        
        df = self.current_df
        
        # Informações básicas
        info = {
            "linhas": df.shape[0],
            "colunas": df.shape[1],
            "colunas_nomes": list(df.columns),
            "tipos_dados": df.dtypes.to_dict(),
            "dados_faltantes": df.isnull().sum().to_dict(),
            "memoria_mb": df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Se tem coluna Class (fraude)
        if 'Class' in df.columns:
            fraud_count = df['Class'].sum()
            info["fraude"] = {
                "total_fraudes": int(fraud_count),
                "taxa_fraude": float((fraud_count / len(df)) * 100),
                "distribuicao": df['Class'].value_counts().to_dict()
            }
        
        # Estatísticas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            info["estatisticas"] = df[numeric_cols].describe().to_dict()
        
        return info
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Responde perguntas básicas sobre os dados"""
        if self.current_df is None:
            return {"error": "Nenhum dataset carregado"}
        
        question_lower = question.lower()
        df = self.current_df
        
        # Respostas baseadas em regras simples
        if "quantas linhas" in question_lower or "quantos registros" in question_lower:
            answer = f"O dataset possui {df.shape[0]:,} linhas (registros)."
            
        elif "quantas colunas" in question_lower:
            answer = f"O dataset possui {df.shape[1]} colunas: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}."
            
        elif "taxa de fraude" in question_lower or "porcentagem de fraude" in question_lower:
            if 'Class' in df.columns:
                fraud_rate = (df['Class'].sum() / len(df)) * 100
                answer = f"A taxa de fraude é {fraud_rate:.2f}% ({df['Class'].sum():,} fraudes em {len(df):,} transações)."
            else:
                answer = "Não foi encontrada coluna 'Class' para análise de fraude."
                
        elif "dados faltantes" in question_lower or "valores ausentes" in question_lower:
            missing = df.isnull().sum()
            total_missing = missing.sum()
            if total_missing == 0:
                answer = "✅ Não há dados faltantes no dataset."
            else:
                cols_with_missing = missing[missing > 0]
                answer = f"Há {total_missing:,} valores ausentes em {len(cols_with_missing)} colunas: " + \
                        ", ".join([f"{col}: {count}" for col, count in cols_with_missing.items()[:5]])
                        
        elif "correlação" in question_lower:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                # Encontrar maior correlação (excluindo diagonal)
                corr_matrix = df[numeric_cols].corr()
                # Remover diagonal e duplicatas
                corr_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                max_corr = corr_matrix.stack().max()
                min_corr = corr_matrix.stack().min()
                answer = f"Maior correlação positiva: {max_corr:.3f}, maior correlação negativa: {min_corr:.3f}."
            else:
                answer = "Não há colunas numéricas suficientes para análise de correlação."
                
        elif "tipos de dados" in question_lower:
            types_count = df.dtypes.value_counts()
            answer = f"Tipos de dados: " + ", ".join([f"{dtype}: {count} colunas" for dtype, count in types_count.items()])
            
        elif "outliers" in question_lower:
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:3]  # Primeiras 3 colunas
            outlier_info = []
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_pct = (outliers / len(df)) * 100
                outlier_info.append(f"{col}: {outliers} ({outlier_pct:.1f}%)")
            answer = f"Outliers detectados: " + ", ".join(outlier_info)
            
        else:
            # Resposta genérica
            info = self.get_basic_info()
            answer = f"""Informações do dataset:
            • {info['linhas']:,} linhas e {info['colunas']} colunas
            • Tamanho: {info['memoria_mb']:.1f} MB
            • Dados faltantes: {sum(info['dados_faltantes'].values()):,}
            """
            if 'fraude' in info:
                answer += f"\n• Taxa de fraude: {info['fraude']['taxa_fraude']:.2f}%"
        
        # Determinar se precisa de visualização
        viz_keywords = ["gráfico", "visualização", "chart", "plot", "distribuição", "correlação", "mostrar"]
        needs_viz = any(keyword in question_lower for keyword in viz_keywords)
        
        result = {
            "answer": answer,
            "visualization": None,
            "data_insights": []
        }
        
        if needs_viz:
            if "correlação" in question_lower:
                result["visualization"] = self.generate_correlation_plot()
            elif "fraude" in question_lower and 'Class' in df.columns:
                result["visualization"] = self.generate_fraud_plot()
        
        # Adicionar ao histórico
        self.conversation_history.append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now()
        })
        
        return result
    
    def generate_correlation_plot(self) -> str:
        """Gera gráfico de correlação"""
        try:
            numeric_df = self.current_df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) < 2:
                return "Erro: Não há colunas numéricas suficientes"
            
            # Limitar a 10 colunas para visualização
            cols_to_plot = numeric_df.columns[:10]
            
            plt.figure(figsize=(10, 8))
            correlation_matrix = numeric_df[cols_to_plot].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       fmt='.2f', square=True)
            plt.title('Matriz de Correlação')
            plt.tight_layout()
            
            # Converter para base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
            img_buffer.seek(0)
            img_str = base64.b64encode(img_buffer.read()).decode()
            plt.close()
            
            return f"data:image/png;base64,{img_str}"
            
        except Exception as e:
            return f"Erro ao gerar gráfico: {str(e)}"
    
    def generate_fraud_plot(self) -> str:
        """Gera gráfico de análise de fraude"""
        try:
            if 'Class' not in self.current_df.columns:
                return "Erro: Coluna 'Class' não encontrada"
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Distribuição de classes
            class_counts = self.current_df['Class'].value_counts()
            axes[0].bar(['Normal', 'Fraude'], [class_counts[0], class_counts[1]], 
                       color=['skyblue', 'salmon'])
            axes[0].set_title('Distribuição: Normal vs Fraude')
            axes[0].set_ylabel('Quantidade')
            
            # Percentual
            total = len(self.current_df)
            percentages = [class_counts[0]/total*100, class_counts[1]/total*100]
            axes[1].pie(percentages, labels=['Normal', 'Fraude'], autopct='%1.1f%%',
                       colors=['skyblue', 'salmon'])
            axes[1].set_title('Proporção de Fraudes')
            
            plt.tight_layout()
            
            # Converter para base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
            img_buffer.seek(0)
            img_str = base64.b64encode(img_buffer.read()).decode()
            plt.close()
            
            return f"data:image/png;base64,{img_str}"
            
        except Exception as e:
            return f"Erro ao gerar gráfico: {str(e)}"

# Teste
if __name__ == "__main__":
    print("🤖 Testando Agente CSV Simples")
    print("=" * 40)
    
    agent = CSVAgentSimple()
    
    # Testar carregamento
    if agent.load_csv("../creditcard.csv"):
        print("\n📊 Informações básicas:")
        info = agent.get_basic_info()
        print(f"Linhas: {info['linhas']:,}")
        print(f"Colunas: {info['colunas']}")
        if 'fraude' in info:
            print(f"Taxa de fraude: {info['fraude']['taxa_fraude']:.2f}%")
        
        print("\n❓ Testando perguntas:")
        questions = [
            "Quantas linhas tem o dataset?",
            "Qual é a taxa de fraude?",
            "Há dados faltantes?"
        ]
        
        for q in questions:
            response = agent.answer_question(q)
            print(f"Q: {q}")
            print(f"R: {response['answer']}\n")
    
    print("✅ Teste concluído!")