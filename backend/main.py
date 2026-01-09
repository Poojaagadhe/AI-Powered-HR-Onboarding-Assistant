import streamlit as st
import os
import shutil
from dotenv import load_dotenv

# --- GOOGLE GEMINI IMPORTS ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load API Key (Try Streamlit Secrets first, then .env)
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    else:
        st.error("⚠️ Google API Key missing! Add it to .env or Streamlit Secrets.")

# 2. Page Config
st.set_page_config(page_title="AI HR Assistant", layout="wide")
st.title("🤖 AI HR Onboarding Assistant")

# 3. Initialize Session State (Memory)
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# 4. Setup Upload Directory
UPLOAD_DIR = "uploaded_docs"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# --- SIDEBAR: Document Upload ---
with st.sidebar:
    st.header("📂 Upload Documents")
    uploaded_file = st.file_uploader("Upload a PDF Policy or Resume", type="pdf")

    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Processing PDF..."):
                try:
                    # Save the file locally
                    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                    with open(file_path, "wb") as buffer:
                        shutil.copyfileobj(uploaded_file, buffer)

                    # Load and Split
                    loader = PyPDFLoader(file_path)
                    documents = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    chunks = text_splitter.split_documents(documents)

                    # Embed and Store
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
                    
                    if st.session_state.vector_store is None:
                        st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
                    else:
                        st.session_state.vector_store.add_documents(chunks)
                        
                    st.success(f"✅ {uploaded_file.name} processed successfully!")
                
                except Exception as e:
                    st.error(f"❌ Error processing file: {e}")

# --- MAIN AREA: Q&A Interface ---
st.subheader("💬 Ask a Question")
user_question = st.text_input("Example: What is the leave policy?")

if user_question:
    if st.session_state.vector_store is None:
        st.warning("⚠️ Please upload and process a document first.")
    else:
        with st.spinner("Thinking..."):
            try:
                # 1. Setup Retriever
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                
                # 2. Setup LLM 
                # Note: Switched to 1.5-flash as 2.5 is sometimes experimental/restricted. 
                # If you have access to 2.5, change it back!
                # We are switching to the standard "gemini-pro" to fix the 404 error
                llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
                
                # 3. Prompt
                template = """Answer the question based only on the following context:
                {context}
                
                Question: {question}
                """
                prompt = ChatPromptTemplate.from_template(template)
                
                # 4. Chain
                chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                answer = chain.invoke(user_question)
                st.write(answer)
                
            except Exception as e:
                st.error(f"❌ Error generating answer: {e}")
