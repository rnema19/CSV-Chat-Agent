# importing basics
import streamlit as st
import os

# importing langchain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.chat_models import ChatOllama

# importing environment variables for security
from dotenv import load_dotenv
load_dotenv()


# importing pinecone DB and setup, didn't use pinecone as pinecone db not available for python 3.12>=

# from pinecone import Pinecone, ServerlessSpec
# from langchain_pinecone import PineconeVectorStore
# pine_api_key = os.environ["PINECONE_API_KEY"]
# pc = Pinecone(api_key=pine_api_key)


# using chromadb for vector storage
from langchain_chroma import Chroma

# giving title to our chatbot
st.title("Chatbot Application with RAG and Vector DB")

# initializing embeddings
ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")

# create vector db
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=ollama_embeddings,
    persist_directory="./chroma_langchain_db",
)

# initialize chat history
if "messages" not in st.session_state:
    # create empty list to initialise
    st.session_state.messages = []
    
    st.session_state.messages.append(SystemMessage("You are an AI-assistant to help in tasks!"))


# appending message to the list as {user:message,assistant:response(message)}
for m in st.session_state.messages:
    if isinstance(m,HumanMessage):
        with st.chat_message("user"):
            st.markdown(m.content)
    elif isinstance(m,AIMessage):
        with st.chat_message("assistant"):
            st.markdown(m.content)

prompt = st.chat_input("Hi! I'm your AI Assistant! How could I help you?")


# when a prompt is submitted
if prompt:
    
    # display user message to the screen with streamlit
    with st.chat_message("user"):
        st.markdown(prompt)
        
        st.session_state.messages.append(HumanMessage(prompt))
    
    # initialize the llm
    llm = ChatOllama(model="gemma3")
    
    # retriever creation
    retriever = vector_store.as_retriever(
        search_type = "similarity_score_threshold",
        search_kwargs = {"k":3,"score_threshold":2}
    )
    
    docs = retriever.invoke(prompt)
    docs_text = "".join(d.page_content for d in docs)
    
    # creating the system prompt
    system_prompt = """You are an AI Assistant. Please answer the queries in short. Try to keep maximum accuracy and try to be concise.
    Context :{context}"""
    
    system_prompt_format = system_prompt.format(context=docs_text)
    
    print ("----SYSTEM_PROMPT-----")
    print(system_prompt_format)
    
    # adding the system prompt to the message history
    st.session_state.messages.append(SystemMessage(system_prompt_format))
    
    # invoke the llm
    result = llm.invoke(st.session_state.messages).content
    
    # adding the response from llm to the screen
    with st.chat_message("assistant"):
            st.markdown(result)
            
            st.session_state.messages.append(AIMessage(result))   
    