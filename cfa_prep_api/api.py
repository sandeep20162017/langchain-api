import os
from flask import Flask, request, jsonify
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

app = Flask(__name__)

# Get the environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')
model_name = os.getenv('MODEL_NAME')
temperature = float(os.getenv('TEMPERATURE', 0.2))  # Default to 0.2 if not set
persist_directory = os.getenv('PERSIST_DIRECTORY', './chroma_db')  # Default to './chroma_db' if not set
prompt_template = os.getenv('PROMPT_TEMPLATE')

# Validate that all required environment variables are set
if not openai_api_key:
    raise ValueError("API key is not valid. Please contact Sandeep.Kanao@gmail.com.")
if not model_name:
    raise ValueError("Model name is not set. Please contact Sandeep.Kanao@gmail.com.")
if not persist_directory:
    raise ValueError("Persist directory is not set. Please contact Sandeep.Kanao@gmail.com.")
if not prompt_template:
    raise ValueError("Prompt template is not set. Please contact Sandeep.Kanao@gmail.com.")

# Load the vector database from the persisted directory
def load_vector_db(persist_directory):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectordb

# Initialize the QA bot
def initialize_qa_bot(vectordb):
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": PROMPT}
    
    # Initialize ChatOpenAI with the specified parameters
    llm = ChatOpenAI(temperature=temperature, model_name=model_name, openai_api_key=openai_api_key)
    
    # Initialize RetrievalQA with the ChatOpenAI instance
    qa_bot = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectordb.as_retriever(search_kwargs={"k": 5}), chain_type_kwargs=chain_type_kwargs, return_source_documents=False)
    return qa_bot

# API endpoint to handle queries
@app.route('/query', methods=['POST'])
def query_handler():
    data = request.json
    query = data.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    vectordb = load_vector_db(persist_directory)
    qa_bot = initialize_qa_bot(vectordb)
    result = qa_bot.run({"query": query})
    return jsonify({"response": result})

if __name__ == "__main__":
    app.run(debug=True)