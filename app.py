import ollama
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# load data

loader = PyPDFLoader("D:\\Users\\Harry Mason\\Downloads\\are-you-ready-guide.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Create Ollama embeddings and vector store
embeddings = OllamaEmbeddings(model="gemma2:2b")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
# Create the retriever
retriever = vectorstore.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
# Define the Ollama LLM function
def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model='gemma2:2b', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

# Define the RAG chain
def rag_chain(question):
    print("retrieving")
    retrieved_docs = retriever.invoke(question)
    print("formatting documents")
    formatted_context = format_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)

# Use the RAG chain
while True:
    user_input = input("Enter a command (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        print("Exiting the program. Stay safe!")
        break
    result = rag_chain(user_input)
    print(result)