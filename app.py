import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. CONFIGURATION & MODELS ---
st.set_page_config(page_title="Enterprise Knowledge RAG", layout="wide")

# Set your API Key here or in a .env file
os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY"

@st.cache_resource
def load_models():
    # Local free embeddings (runs on CPU)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Gemini Free Tier LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
    return embeddings, llm

embeddings_model, llm_model = load_models()

# --- 2. DATA INGESTION & ADVANCED INDEXING ---
def process_document(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Sentence-Window Logic: Small chunks for retrieval, larger context for synthesis
    # We use Recursive splitter to simulate granular indexing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(documents)
    
    # Create Vector Store (Dense Retrieval)
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings_model,
        collection_name="enterprise_docs"
    )
    
    # Create BM25 (Keyword/Sparse Retrieval)
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 3
    
    # --- 3. MULTI-RETRIEVER SYSTEM ---
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vectorstore.as_retriever(search_kwargs={"k": 5}), bm25_retriever],
        weights=[0.6, 0.4] # Give more weight to semantic search
    )
    
    # --- 4. RE-RANKING LAYER (FlashRank) ---
    # This filters the top 10 documents down to the most relevant 3
    compressor = FlashrankRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=ensemble_retriever
    )
    
    return compression_retriever

# --- 5. RAG CHAIN SETUP ---
def get_rag_chain(retriever):
    system_prompt = (
        "You are an enterprise assistant. Use the following pieces of retrieved context "
        "to answer the question. If you don't know the answer, say you don't know. "
        "Provide citations for your sources.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm_model, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

# --- 6. STREAMLIT UI ---
st.title("📑 Advanced Multi-Source Enterprise RAG")
st.sidebar.header("Upload Documents")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    # Save file locally
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Analyzing and Indexing Document..."):
        retriever = process_document("temp.pdf")
        rag_chain = get_rag_chain(retriever)
    st.sidebar.success("Document Indexed!")

    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Ask a question about your knowledge base..."):
        st.session_state.messages.append({"role": "human", "content": query})
        with st.chat_message("human"):
            st.markdown(query)

        with st.chat_message("assistant"):
            response = rag_chain.invoke({"input": query})
            answer = response["answer"]
            st.markdown(answer)
            
            # Show Citations/Sources
            with st.expander("View Source Chunks (Reranked)"):
                for doc in response["context"]:
                    st.write(f"- {doc.page_content[:200]}... (Page: {doc.metadata.get('page')})")
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Please upload a PDF document to begin.")
