import streamlit as st
import time
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

st.title("RAG Application built on Gemini Model")

# Add PDF upload functionality
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Initialize session state for vectorstore
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.file_processed = False
    st.session_state.current_file = None

if uploaded_file is not None:
    # Check if we need to process a new file
    if st.session_state.current_file != uploaded_file.name:
        with st.spinner('Processing PDF file...'):
            # Save the uploaded file temporarily
            temp_file_path = f"temp_{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load and process the PDF
            loader = PyPDFLoader(temp_file_path)
            data = loader.load()
            
            # Split the documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
            docs = text_splitter.split_documents(data)
            
            # Create vector store
            st.session_state.vectorstore = FAISS.from_documents(
                documents=docs, 
                embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            )
            
            # Update session state
            st.session_state.file_processed = True
            st.session_state.current_file = uploaded_file.name
            
            # Clean up
            os.remove(temp_file_path)
        
        st.success(f"Processed {uploaded_file.name}")
else:
    if not st.session_state.file_processed:
        st.warning("Please upload a PDF file to provide context for your questions.")

# Only show the query input if a file has been processed
if st.session_state.file_processed:
    # Create retriever
    retriever = st.session_state.vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 10}
    )
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None
    )
    
    # Set up the prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    # Create chat input
    query = st.chat_input("Ask a question about the PDF content:")
    
    # Display conversation history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Process the query
    if query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display user message
        with st.chat_message("user"):
            st.write(query)
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Create the chain
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                
                # Get response
                response = rag_chain.invoke({"input": query})
                answer = response["answer"]
                
                # Display the answer
                st.write(answer)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})