import streamlit as st
import os
import pandas as pd

# import dotenv files
from dotenv import load_dotenv
load_dotenv()

# importing langchain modules
from langchain_experimental.agents import create_csv_agent
from langchain_ollama import OllamaLLM, OllamaEmbeddings, ChatOllama
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

def agent(file_path):
    llm = ChatOllama(model="qwen3:1.7b", temperature=0.5)
    agent_executor = create_csv_agent(
        llm=llm,
        path=file_path,
        verbose=True,
        allow_dangerous_code = True
    )
    return agent_executor

def main():
    st.set_page_config(page_title="Chat App")
    st.header("Chat with your CSV Files")
    
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
    
    user_file = st.file_uploader("Upload your CSV file",type=".csv")
    
    if user_file:
        file_details = {"FileName": user_file.name, "FileType": user_file.type, "FileSize": user_file.size}
        st.write(file_details)
        
        # Save the uploaded file to disk
        file_path = os.path.join("documents",user_file.name)
        with open(file_path, "wb") as f:
            f.write(user_file.getbuffer())
        
        df = pd.read_csv(file_path)
        st.dataframe(df)
        
        user_question = st.text_input("Ask me anything about your file!") 
        if user_question:
            with st.spinner(text="In progress..."):
                agent_executor = agent(file_path)              
                response = agent_executor.invoke(input=user_question)
                # If response is a dict with 'output'
                if isinstance(response, dict) and "output" in response:
                    response_text = response["output"]
                else:
                    response_text = response
                # Save history in session and in a file
                st.session_state.messages.append((user_question, response_text))
                
                with open("chat_logs/chat_history.txt", "a+") as log_file:
                    log_file.write(f"User: {user_question}\n")
                    log_file.write(f"Bot: {response_text}\n\n")

                st.write(response_text)

        # Display conversation history
        if st.session_state.messages:
            st.subheader("Chat History")
            for i, entry in enumerate(st.session_state.messages, 1):
                if isinstance(entry, tuple) and len(entry) == 2:
                    q, a = entry
                    st.markdown(f"**Q{i}:** {q}")
                    st.markdown(f"**A{i}:** {a}")
                else:
                    st.warning(f"Skipping malformed history entry: {entry}")
                    
        if st.button("Clear Chat History"):
            st.session_state.history = []
            with open("chat_logs/chat_history.txt", "w+") as log_file:
                log_file.write("")  # clear the file
            st.success("Chat history cleared.")
            
if __name__=="__main__":
    main()
    
