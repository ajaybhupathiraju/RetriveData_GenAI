import dotenv
import streamlit as st
from langchain_community.llms.openai import OpenAIChat
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import OpenAI,ChatOpenAI
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS



load_dotenv()
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["GORQ_API_KEY"]=os.getenv("GORQ_API_KEY")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

# Load data, Splitting data, Embeddings, store into vectordatabase, Retrival

# loading
text_loader = TextLoader("./speech.txt")
document = text_loader.load()

# splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=0)
docs = text_splitter.split_documents(document)

# embedding
openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=1024)

# convert text/documents into embedding vectors and store into vector store FAISS database.
faiss_db = FAISS.from_documents(docs,openai_embeddings)
faiss_db.save_local("./faiss_db")

