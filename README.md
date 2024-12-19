![LangChain](https://img.shields.io/badge/LangChain-%2300599C.svg?style=flat&logo=langchain&logoColor=white) 
![Streamlit](https://img.shields.io/badge/Streamlit-%23FF4B4B.svg?style=flat&logo=streamlit&logoColor=white) 
![FAISS](https://img.shields.io/badge/FAISS-%234285F4.svg?style=flat&logo=faiss&logoColor=white) 
![Groq](https://img.shields.io/badge/Groq-%23FFAA00.svg?style=flat&logo=groq&logoColor=white) 
![Ollama](https://img.shields.io/badge/Ollama-%2300555A.svg?style=flat&logo=ollama&logoColor=white) 

# Retrieval-Augmented Generation (RAG) Workflow with LangChain, Streamlit, and Groq

This repository demonstrates how to implement a **Retrieval-Augmented Generation (RAG)** workflow using LangChain, Streamlit, and the Groq API. The app retrieves relevant document chunks based on a user query and uses a language model to generate an accurate response grounded in the retrieved context.

## Features
- **🔗 LangChain**: Provides modular tools for document loading, splitting, and retrieval.
- **🧠 Groq LLM**: Integrates Groq’s advanced language models for reasoning and generation.
- **💾 FAISS**: Efficient vector store for document similarity search.
- **📜 Streamlit**: Interactive web interface for querying and results display.
- **📚 Ollama Embeddings**: Generates embeddings for vector similarity search.

---

## Workflow

1. **Document Loading**:
   - Uses **`WebBaseLoader`** to fetch documents from a specified URL.

2. **Text Splitting**:
   - Splits documents into manageable chunks (1000 characters with 200-character overlap).

3. **Embedding & Storage**:
   - Converts text chunks into embeddings using **OllamaEmbeddings**.
   - Stores the embeddings in a **FAISS** vector database for efficient retrieval.

4. **Question Answering**:
   - Retrieves relevant chunks using FAISS.
   - Passes the chunks and user query to a **Groq language model**.

5. **Interactive App**:
   - Streamlit provides a user-friendly interface for querying and displaying results.

---

## Code Overview

### Importing Libraries
```python
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv
```
- **Streamlit**: For building the web interface.
- **Groq API**: To connect with Groq’s LLM.
- **LangChain Tools**: For document processing, embeddings, and chains.
- **FAISS**: Efficient vector storage for retrieval tasks.

### Loading Documents
```python
st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
st.session_state.docs = st.session_state.loader.load()
```
- **WebBaseLoader**: Downloads documents from the given URL.

### Splitting and Embedding
```python
st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
st.session_state.embeddings = OllamaEmbeddings()
st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
```
- Splits documents into manageable chunks.
- Embeds the chunks and stores them in a FAISS vector database.

### Defining the Language Model
```python
llm = ChatGroq(groq_api_key=os.environ['GROQ_API_KEY'], model_name="mixtral-8x7b-32768")
```
- Connects to Groq’s LLM using an API key.

### Prompt and Retrieval Chain
```python
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    <context>
    {context}
    <context>
    Question: {input}
    """
)
retriever = st.session_state.vectors.as_retriever()
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)
```
- Creates a template to structure how the LLM receives context and user queries.
- Links the retriever and the document chain to form the RAG workflow.

### User Interaction
```python
prompt = st.text_input("Input your prompt here")
if prompt:
    response = retrieval_chain.invoke({"input": prompt})
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for doc in response["context"]:
            st.write(doc.page_content)
            st.write("--------------------------------")
```
- Accepts a query from the user.
- Retrieves relevant document chunks and generates a response.
- Displays the answer and the retrieved document chunks.

---

## Tools and Frameworks Used

| Tool/Framework       | Badge                                                                 |
|----------------------|----------------------------------------------------------------------|
| LangChain            | ![LangChain](https://img.shields.io/badge/LangChain-%2300599C.svg?style=flat&logo=langchain&logoColor=white) |
| Streamlit            | ![Streamlit](https://img.shields.io/badge/Streamlit-%23FF4B4B.svg?style=flat&logo=streamlit&logoColor=white) |
| FAISS                | ![FAISS](https://img.shields.io/badge/FAISS-%234285F4.svg?style=flat&logo=faiss&logoColor=white) |
| Groq API             | ![Groq](https://img.shields.io/badge/Groq-%23FFAA00.svg?style=flat&logo=groq&logoColor=white) |
| Ollama Embeddings    | ![Ollama](https://img.shields.io/badge/Ollama-%2300555A.svg?style=flat&logo=ollama&logoColor=white) |

---

## How It Works
1. **Preprocessing**:
   - Loads documents, splits them into chunks, and embeds them in a FAISS vector store.
2. **User Query**:
   - The user submits a question via the Streamlit interface.
3. **Context Retrieval**:
   - Relevant chunks are retrieved from the FAISS vector store.
4. **Answer Generation**:
   - The Groq LLM generates an answer based on the retrieved context.
5. **Results Display**:
   - The app displays the generated answer and the relevant document chunks.

