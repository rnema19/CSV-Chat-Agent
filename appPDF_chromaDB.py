# importing basics
import streamlit as st
import os

# importing langchain
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOllama
from pypdf import PdfReader

# importing environment variables for security
from dotenv import load_dotenv
load_dotenv()

# import google genai requirement and api setup
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
google_api_key = os.environ["GOOGLE_API_KEY"]


# import vector db
from langchain_chroma import Chroma

# load your documents and append the text into readable format
def get_pdf_text(docs):
    text=""
    for doc in docs:
        pdf_reader = PdfReader(doc)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

# split your text chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap = 200)
    chunks = text_splitter.split_text(text=text)
    return chunks


# create vector db and embeddings
def vector_storage(data):
    google_embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=google_embeddings,
    persist_directory="./chroma_langchain_db"
    )
    vector_store.add_texts(data)
    with open("./chroma_langchain_db/chunks.txt", "w", encoding="utf-8") as f:
        for chunk in data:
            f.write(chunk + "\n---\n")
    return vector_store

def chain_creation():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer
    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    # prompt = ChatPromptTemplate(prompt_template,input_variables=["context","question"])
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatOllama(model="gemma3",temperature=0)
    chain = load_qa_chain(llm=llm,chain_type="stuff",prompt=prompt)
    return chain

# 
def take_input(user_question):
    google_embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=google_embeddings,
    persist_directory="./chroma_langchain_db"
    )    
    docs = vector_store.similarity_search(input)
    
    chain = chain_creation()
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)    
    print(response)
    
    st.write("Reply: ", response["output_text"])
    
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using AI-Open Source💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        take_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_storage(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
