import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# --- GOOGLE GEMINI IMPORTS ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load API Key
load_dotenv()

app = FastAPI(title="AI HR Assistant (Gemini Powered)")

# --- GLOBAL VARIABLES ---
vector_store = None
UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "HR AI Assistant is ready. Upload a PDF!"}

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    global vector_store
    
    try:
        # 1. Save File
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Load PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # 3. Split Text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        # 4. Embed & Store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        
        if vector_store is None:
            vector_store = FAISS.from_documents(chunks, embeddings)
        else:
            vector_store.add_documents(chunks)

        return {"status": "Success", "filename": file.filename, "message": "Document processed successfully!"}
    
    except Exception as e:
        return {"status": "Error", "message": str(e)}

@app.post("/ask")
def ask_question(request: QueryRequest):
    global vector_store
    if vector_store is None:
        return {"answer": "⚠️ Error: Please upload a document first (Vector Store is empty)."}

    try:
        # 1. Setup Retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # 2. Setup LLM (Trying the newest model on your list)
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        
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
        
        print(f"🤖 Asking Gemini 2.5: {request.question}")
        answer = chain.invoke(request.question)
        return {"answer": answer}

    except Exception as e:
        print(f"❌ ERROR IN ASK: {e}")
        return {"answer": f"⚠️ SYSTEM ERROR: {str(e)}"}
