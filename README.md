# Chatbot para Consulta de Manuais da Empresa

Este projeto é um **chatbot inteligente**, desenvolvido com **Python** e **Streamlit**, capaz de responder perguntas relacionadas aos **manuais internos da empresa**.  
O sistema utiliza processamento de linguagem natural (NLP), embeddings vetoriais e busca semântica para encontrar informações relevantes nos PDFs indexados.

---

##  Funcionalidades

- **Indexação automática de PDFs** presentes na pasta `./manuais/`
- **Busca semântica** usando ChromaDB + SentenceTransformer Embeddings
- **Geração de respostas inteligentes** utilizando o modelo *Gemini* da Google (via LangChain)
- **Interface moderna de chat** integrada ao Streamlit
- Armazena histórico das conversas
- Banco vetorial persistente (não precisa reindexar sempre)

---

## Tecnologias Utilizadas

- **Python 3.10+**
- **Streamlit**
- **LangChain**
- **Google Gemini API**
- **ChromaDB**
- **SentenceTransformer Embeddings**
- **PyPDFLoader**
- **YAML**

---

## Interface
<img width="1910" height="905" alt="image" src="https://github.com/user-attachments/assets/17c68a4d-8483-44db-83e8-eaaa9349236d" />

1. **Clone este repositório:**  
   ```bash
   git clone https://github.com/seu-usuario/ChatBot-com-Manuais-de-Empresas.git

