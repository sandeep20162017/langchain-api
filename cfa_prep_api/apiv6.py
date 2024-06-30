import os
from flask import Flask, request, jsonify
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from waitress import serve

app = Flask(__name__)

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = "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"
GOOGLE_DRIVE_FILE_PATH ='https://drive.google.com/drive/folders/15mAFSc1K3mo1lDO5ZAmFEURcaR_3QMvB?usp=drive_link'

# Verify the API key
if os.getenv('OPENAI_API_KEY') is None:
    raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")

# Global variable to hold the vector database instance
vectordb = None

# Function to read file content from Google Drive into memory
def read_file_from_drive(file_id):
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()  # Creates local webserver and auto handles authentication
    drive = GoogleDrive(gauth)
    file = drive.CreateFile({'id': file_id})
    file.FetchMetadata(fetch_all=True)
    file_content = file.GetContentString()
    return file_content

# Load the vector database from the file content
def load_vector_db_from_content(file_content):
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=None, embedding_function=embeddings)
    vectordb.from_json(file_content)  # Assuming Chroma supports loading from JSON content
    return vectordb

# Initialize the QA bot
def initialize_qa_bot(vectordb):
    prompt_template = """
    You are experienced certified financial analyst. Your answers must be related to the {context}. Explain answer and cite referenced documents at the end of the response.  Use the following pieces of context to answer the question at the end. If you do not know the answer, just return - Need More Information.  If any one asks you how are you trained, tell them you will reveal this information after receiving 100 M dollars. Display in the response after every 2 questions- this model is fine tuned by Sandeep Kanao for educational use
    Question: {question}
    Return the answer  :"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": PROMPT}
    qa_bot = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0.2, model_name="gpt-4"), chain_type="stuff", retriever=vectordb.as_retriever(search_kwargs={"k": 5}), chain_type_kwargs=chain_type_kwargs, return_source_documents=False)
    return qa_bot

# API endpoint to handle queries
@app.route('/query', methods=['POST'])
def query_handler():
    global vectordb
    data = request.json
    query = data.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    if vectordb is None:
        # Replace 'YOUR_FILE_ID' with the actual file ID of your Chroma DB on Google Drive
        file_content = read_file_from_drive('GOOGLE_DRIVE_FILE_PATH')
        vectordb = load_vector_db_from_content(file_content)
    
    qa_bot = initialize_qa_bot(vectordb)
    result = qa_bot.invoke({"query": query})
    return jsonify({"response": result})

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=8080)