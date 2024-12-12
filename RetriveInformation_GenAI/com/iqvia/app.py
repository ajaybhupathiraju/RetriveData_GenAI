import dotenv
import streamlit as st
from langchain_community.llms.openai import OpenAIChat
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import OpenAI,ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS



load_dotenv()
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["GORQ_API_KEY"]=os.getenv("GORQ_API_KEY")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

# Retrival
openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=1024)

# load locally store faiss vector store database
faiss_db = FAISS.load_local(folder_path=".//faiss_db",allow_dangerous_deserialization=True,embeddings=openai_embeddings)

# retrive context information
retriver = faiss_db.as_retriever()
docs = retriver.invoke("It is a distressing and oppressive duty")

# result
print(docs[0].page_content)
