import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
import glob
import yaml
import shutil

# Configurações do Streamlit
st.set_page_config(page_title="Chat com Manuais da Empresa", layout="wide")

# Carregar configurações
@st.cache_resource # Cache para não recarregar em cada interação
def load_config():
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    # Configuração
    GEMINI_KEY = config['KEY']
    PASTA_PDFS = "./manuais"
    CHROMA_PATH = "./chroma_db"

    # Inicializar
    gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_KEY)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    return gemini, embeddings, CHROMA_PATH, PASTA_PDFS


def indexar_pdfs(embeddings, CHROMA_PATH):
    # Limpar banco existente (opcional)
    if os.path.exists(CHROMA_PATH):
        st.info("Removendo banco de dados antigo...")
        shutil.rmtree(CHROMA_PATH)
    
    # Carregar PDFs
    documentos = []
    for arquivo in glob.glob("./manuais/*.pdf"):
        st.write(f"Carregando {os.path.basename(arquivo)}")
        loader = PyPDFLoader(arquivo)
        pages = loader.load()
        
        # Adicionar metadados personalizados
        for page in pages:
            page.metadata['manual'] = os.path.basename(arquivo).replace('.pdf', '')
            documentos.append(page)
    
    if not documentos:
        st.warning("Nenhum PDF encontrado na pasta ./manuais/")
        return
    
    # Criar vector store no ChromaDB
    st.info(f"Processando {len(documentos)} documentos...")
    vectorstore = Chroma.from_documents(
        documents=documentos,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name="manuais-empresa"
    )
    
    st.success(f"Indexados {len(documentos)} documentos no ChromaDB!")


def consultar(pergunta, embeddings, gemini, CHROMA_PATH):
    # Verificar se banco existe
    if not os.path.exists(CHROMA_PATH):
        return "Execute 'Indexar PDFs' primeiro para criar a base de conhecimento!"
    
    # Conectar ao ChromaDB existente
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="manuais-empresa"
    )
    
    # Buscar documentos similares
    results = vectorstore.similarity_search_with_score(pergunta, k=3)
    
    # Verificar se encontrou resultados
    if not results:
        return "Não encontrei informações relevantes nos manuais."
    
    # Montar contexto
    contexto = ""
    for doc, score in results:
        manual = doc.metadata.get('manual', 'Desconhecido')
        texto = doc.page_content[:1000]  # Limitar texto
        # ChromaDB usa distância (quanto menor, melhor), converter para similaridade
        similaridade = 1 - score
        contexto += f"Manual {manual} (relevância: {similaridade:.2f}):\n{texto}\n\n"
    
    # Gerar resposta
    prompt = f"""Com base nos manuais da empresa abaixo, responda de forma clara e prática:

{contexto}

Pergunta: {pergunta}

Resposta baseada nos manuais:"""
    
    return gemini.invoke(prompt).content


# Interface Streamlit
st.title("Chat com Manuais da Empresa")

# Carregar componentes
gemini, embeddings, CHROMA_PATH, PASTA_PDFS = load_config()

# Sidebar
with st.sidebar:
    st.header("Configurações")
    
    if st.button("Indexar PDFs", type="primary"): #ação principal
        with st.spinner("Indexando PDFs..."):
            indexar_pdfs(embeddings, CHROMA_PATH)
    
    st.divider() # Cria uma linha horizontal
    
    # Mostrar status do banco
    if os.path.exists(CHROMA_PATH):
        st.success("Banco de dados ChromaDB ativo")
    else:
        st.warning("Banco não indexado")
    
    # Listar PDFs disponíveis
    pdf_files = glob.glob("./manuais/*.pdf")
    st.write(f"PDFs encontrados: {len(pdf_files)}")
    
    if pdf_files:
        with st.expander("Ver arquivos"):
            for pdf in pdf_files:
                st.text(f" - {os.path.basename(pdf)}")

# Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensagens anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input do usuário
if prompt := st.chat_input("Digite sua pergunta..."):
    # Adicionar mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Gerar e mostrar resposta
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            resposta = consultar(prompt, embeddings, gemini, CHROMA_PATH)
        st.markdown(resposta)
    
    # Adicionar resposta ao histórico
    st.session_state.messages.append({"role": "assistant", "content": resposta})