"""
Interface Streamlit para Agente CSV I2A2
Usando versão simplificada + Ollama híbrido
"""

import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime
import requests

# Adicionar diretório atual ao path para imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from csv_agent_hybrid import CSVAgentHybrid

# Configuração da página
st.set_page_config(
    page_title="Agente CSV I2A2",
    page_icon="🤖",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background: #f0f8ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1e88e5;
    }
</style>
""", unsafe_allow_html=True)

def check_ollama():
    """Verifica se Ollama está disponível"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def query_ollama_simple(question):
    """Consulta simples ao Ollama"""
    try:
        payload = {
            "model": "llama3.1",
            "prompt": f"Responda brevemente em português: {question}",
            "stream": False,
            "options": {"temperature": 0.7}
        }
        response = requests.post("http://localhost:11434/api/generate", 
                               json=payload, timeout=15)
        if response.status_code == 200:
            return response.json()["response"]
    except:
        return None

# Header
st.markdown("""
<div class="header">
    <h1>🤖 Agente Autônomo CSV - I2A2</h1>
    <p>Instituto de Inteligência Artificial Aplicada</p>
    <p>Especializado em Detecção de Fraudes e Análise Exploratória</p>
    <p>Criado por **Jacqueline Lemos**</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Status
    ollama_status = check_ollama()
    if ollama_status:
        st.success("✅ Ollama Online")
    else:
        st.warning("⚠️ Ollama Offline")
    
    st.header("📁 Upload CSV")
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")
    
    st.header("🎯 Perguntas Exemplo")
    example_questions = [
        "Quantas linhas tem o dataset?",
        "Qual é a taxa de fraude?",
        "Há dados faltantes?",
        "Mostre a correlação entre variáveis",
        "Análise de outliers",
        "Distribuição das fraudes"
    ]

# Inicializar agente
@st.cache_resource
def get_agent():
    openai_key = os.getenv('OPENAI_API_KEY')  # Ou colocar chave direta aqui
    return CSVAgentHybrid(openai_api_key=openai_key)

agent = get_agent()

# Inicializar session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'dataset_loaded' not in st.session_state:
    st.session_state.dataset_loaded = False

# Upload e processamento
if uploaded_file is not None:
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if not st.session_state.dataset_loaded:
        with st.spinner("🔄 Carregando dataset..."):
            success = agent.load_csv(temp_path)
            if success:
                st.session_state.dataset_loaded = True
                st.rerun()
        
        # Limpar arquivo temporário
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Se dataset carregado
if st.session_state.dataset_loaded and agent.current_df is not None:
    
    # Métricas principais
    df = agent.current_df
    info = agent.get_basic_info()
    
    st.success("✅ Dataset carregado com sucesso!")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Registros", f"{info['linhas']:,}")
    
    with col2:
        st.metric("📋 Colunas", info['colunas'])
    
    with col3:
        if 'fraude' in info:
            st.metric("🚨 Fraudes", f"{info['fraude']['total_fraudes']:,}")
        else:
            st.metric("❓ Dados Faltantes", f"{sum(info['dados_faltantes'].values()):,}")
    
    with col4:
        if 'fraude' in info:
            st.metric("📈 Taxa Fraude", f"{info['fraude']['taxa_fraude']:.2f}%")
        else:
            st.metric("💾 Tamanho", f"{info['memoria_mb']:.1f} MB")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat Inteligente", "📊 Análise Rápida", "📋 Dados"])
    
    with tab1:
        st.header("🤖 Chat com o Agente")
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Mostrar histórico
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    if "image" in message:
                        st.image(message["image"])
            
            # Input do usuário
            if prompt := st.chat_input("Faça sua pergunta sobre os dados..."):
                
                # Adicionar pergunta do usuário
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.write(prompt)
                
                # Processar resposta
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Analisando..."):
                        
                        # Tentar resposta inteligente primeiro (se Ollama disponível)
                        ai_response = None
                        if ollama_status and len(prompt.split()) < 10:  # Perguntas curtas
                            ai_response = query_ollama_simple(prompt)
                        
                        # Resposta do agente
                        agent_response = agent.answer_question(prompt)
                        
                        # Combinar respostas
                        if ai_response and "dados" not in prompt.lower():
                            final_response = f"🧠 **Análise IA:** {ai_response}\n\n📊 **Análise dos Dados:** {agent_response['answer']}"
                        else:
                            final_response = agent_response['answer']
                        
                        st.write(final_response)
                        
                        # Mostrar visualização se houver
                        if agent_response.get('visualization'):
                            st.image(agent_response['visualization'])
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": final_response,
                                "image": agent_response['visualization']
                            })
                        else:
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": final_response
                            })
                
                # Insights automáticos
                if agent_response.get('data_insights'):
                    st.info("💡 " + " | ".join(agent_response['data_insights']))
        
        # Botões de exemplo na sidebar
        with st.sidebar:
            st.subheader("⚡ Teste Rápido")
            for i, question in enumerate(example_questions):
                if st.button(question, key=f"example_{i}", use_container_width=True):
                    # Simular input do usuário
                    st.session_state.messages.append({"role": "user", "content": question})
                    
                    # Processar resposta
                    agent_response = agent.answer_question(question)
                    
                    # Resposta combinada
                    if ollama_status and len(question.split()) < 8:
                        ai_response = query_ollama_simple(question)
                        if ai_response:
                            final_response = f"🧠 **IA:** {ai_response}\n\n📊 **Dados:** {agent_response['answer']}"
                        else:
                            final_response = agent_response['answer']
                    else:
                        final_response = agent_response['answer']
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": final_response
                    })
                    
                    st.rerun()
    
    with tab2:
        st.header("📊 Análises Rápidas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔥 Gerar Correlação", use_container_width=True):
                with st.spinner("Gerando correlação..."):
                    viz = agent.generate_correlation_plot()
                    if viz.startswith("data:image"):
                        st.image(viz, caption="Matriz de Correlação")
                    else:
                        st.error(viz)
        
        with col2:
            if 'fraude' in info and st.button("🚨 Análise de Fraude", use_container_width=True):
                with st.spinner("Analisando fraudes..."):
                    viz = agent.generate_fraud_plot()
                    if viz.startswith("data:image"):
                        st.image(viz, caption="Análise de Fraudes")
                    else:
                        st.error(viz)
        
        # Informações detalhadas
        st.subheader("📈 Estatísticas Detalhadas")
        
        if 'fraude' in info:
            st.markdown(f"""
            **Análise de Fraudes:**
            - Total de transações: {info['linhas']:,}
            - Fraudes detectadas: {info['fraude']['total_fraudes']:,}
            - Taxa de fraude: {info['fraude']['taxa_fraude']:.3f}%
            - Transações normais: {info['fraude']['total_fraudes'] - info['linhas']:,}
            """)
        
        if 'estatisticas' in info:
            st.subheader("🔢 Estatísticas Numéricas")
            stats_df = pd.DataFrame(info['estatisticas'])
            st.dataframe(stats_df)
    
    with tab3:
        st.header("📋 Informações do Dataset")
        
        # Overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ℹ️ Informações Gerais")
            st.write(f"**Arquivo:** {uploaded_file.name if uploaded_file else 'creditcard.csv'}")
            st.write(f"**Linhas:** {info['linhas']:,}")
            st.write(f"**Colunas:** {info['colunas']}")
            st.write(f"**Tamanho:** {info['memoria_mb']:.1f} MB")
            st.write(f"**Dados faltantes:** {sum(info['dados_faltantes'].values()):,}")
        
        with col2:
            st.subheader("🏷️ Tipos de Dados")
            types_df = pd.DataFrame(list(info['tipos_dados'].items()), 
                                  columns=['Coluna', 'Tipo'])
            st.dataframe(types_df, height=300)
        
        # Amostra dos dados
        st.subheader("👀 Amostra dos Dados")
        sample_size = st.slider("Linhas para mostrar:", 5, 50, 10)
        st.dataframe(df.head(sample_size))
        
        # Conclusões automáticas
        st.subheader("🧠 Conclusões Automáticas")
        
        conclusions = []
        if 'fraude' in info:
            fraud_rate = info['fraude']['taxa_fraude']
            if fraud_rate < 0.5:
                conclusions.append("✅ Dataset altamente desbalanceado - típico de detecção de fraudes")
            if fraud_rate > 5:
                conclusions.append("⚠️ Alta taxa de fraude - investigar padrões")
        
        if sum(info['dados_faltantes'].values()) == 0:
            conclusions.append("✅ Dataset limpo - sem dados faltantes")
        
        if info['colunas'] > 20:
            conclusions.append("📊 Dataset complexo - muitas features disponíveis")
        
        if info['linhas'] > 100000:
            conclusions.append("📈 Big Data - dataset grande para análises robustas")
        
        for conclusion in conclusions:
            st.success(conclusion)

else:
    # Página inicial
    st.markdown("""
    ## 👋 Bem-vindo ao Agente Autônomo CSV I2A2!
    
    ### 🎯 Funcionalidades:
    - **Análise Exploratória Automática**
    - **Detecção de Fraudes**
    - **Visualizações Inteligentes**
    - **Chat com IA** (quando disponível)
    - **Insights Automáticos**
    
    ### 🚀 Para começar:
    1. Faça upload de um arquivo CSV na barra lateral
    2. Aguarde o carregamento e análise
    3. Faça perguntas sobre seus dados
    4. Explore visualizações automáticas
    
    ### 📊 Exemplo com Dataset de Fraudes:
    O agente foi testado com um dataset de 284,807 transações de cartão de crédito,
    identificando uma taxa de fraude de 0.17% (492 casos).
    """)
    
    # Carregar dataset padrão
    if st.button("🔄 Carregar Dataset de Exemplo (Fraudes)", use_container_width=True):
        example_path = "../creditcard.csv"
        if os.path.exists(example_path):
            with st.spinner("Carregando dataset de exemplo..."):
                success = agent.load_csv(example_path)
                if success:
                    st.session_state.dataset_loaded = True
                    st.success("✅ Dataset de exemplo carregado!")
                    st.rerun()
                else:
                    st.error("❌ Erro ao carregar dataset de exemplo")
        else:
            st.error("❌ Dataset de exemplo não encontrado")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    🤖 Agente Autônomo CSV I2A2 | Desenvolvido para detecção de fraudes e análise de dados
</div>
""", unsafe_allow_html=True)
