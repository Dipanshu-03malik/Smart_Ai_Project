import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import time

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    the provided context just say, "answer is not available in the context", don't provide the wrong answer.\n\n
    Context:\n {context}?\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.search(query=user_question, search_type="similarity", k=10)  # Retrieve the top 10 documents from the index
    chain = get_conversational_chain()
    
    response = chain.invoke({"input_documents": docs, "question": user_question})
    st.write("Reply: ", response["output_text"])

# Chat with PDF functionality
def chat_with_pdf():
    st.markdown("<h2 style='color:#fc1008'>Chat with PDF using Gemini</h2>", unsafe_allow_html=True)

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Done")

# Function to scrape webpage content using Selenium and BeautifulSoup
def get_webpage_text(url):
    # Automatically manage ChromeDriver using ChromeDriverManager
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)

    # Load the webpage
    driver.get(url)

    # Allow the page to load
    time.sleep(5)

    # Get the page content
    page_source = driver.page_source
    driver.quit()

    # Parse the content using BeautifulSoup
    soup = BeautifulSoup(page_source, 'html.parser')
    text = soup.get_text(separator="\n")
    
    return text

# Chat with Webpage functionality
def chat_with_webpage():
    st.markdown("<h2 style='color:#fc1008'>Chat with Webpage using Gemini</h2>", unsafe_allow_html=True)

    webpage_url = st.text_input("Enter the URL of the webpage")

    if webpage_url and st.button("Submit & Process Webpage"):
        with st.spinner("Scraping webpage..."):
            raw_text = get_webpage_text(webpage_url)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Webpage content processed successfully")

    user_question = st.text_input("Ask a Question from the Webpage")

    if user_question:
        user_input(user_question)

# Chat with AI functionality
def chat_with_ai():
    st.markdown("<h2 style='color:#fc1008'>Chat with AI using Gemini</h2>", unsafe_allow_html=True)

    user_question = st.text_input("Ask a Question to the AI")

    if user_question:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel("gemini-pro")

        chat_history = []
        if 'chat_history' in st.session_state:
            chat_history = st.session_state['chat_history']

        chat = model.start_chat(history=chat_history)

        def get_gemini_response(question):
            response = chat.send_message(question, stream=True)
            return response

        response = get_gemini_response(user_question)

        response_text = ""
        for chunk in response:
            response_text += chunk.text

        st.write("Reply: ", response_text)

        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        st.session_state['chat_history'].append({"role": "You", "content": user_question})
        st.session_state['chat_history'].append({"role": "Bot", "content": response_text})

        st.subheader("The Chat History is")
        for message in st.session_state['chat_history']:
            st.write(f"{message['role']}: {message['content']}")

# Main function
def main():
    st.set_page_config("Chat Options", layout="wide")

    # Sidebar control via session state
    if 'sidebar_state' not in st.session_state:
        st.session_state['sidebar_state'] = 'collapsed'

    # Custom CSS to change sidebar width dynamically
    st.markdown(
        """
        <style>
            .sidebar-style {
                transition: width 0.5s;
                background-color: #fc1008;
                color: white;
                padding: 20px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Toggle sidebar button
    if st.button("☰ Smart Ai", key="toggle_sidebar"):
        if st.session_state['sidebar_state'] == 'collapsed':
            st.session_state['sidebar_state'] = 'expanded'
        else:
            st.session_state['sidebar_state'] = 'collapsed'

    # Sidebar appears in expanded or collapsed state
    if st.session_state['sidebar_state'] == 'expanded':
        with st.sidebar:
            st.markdown("<h1 style='color:red'>Menu</h1>", unsafe_allow_html=True)
            options = ["Chat with PDF", "Chat with Webpage", "Chat with AI"]
            selected_option = st.selectbox("Select an option", options)

            if selected_option == "Chat with PDF":
                chat_with_pdf()
            elif selected_option == "Chat with Webpage":
                chat_with_webpage()
            elif selected_option == "Chat with AI":
                chat_with_ai()

# Run the app
if __name__ == "__main__":
    main()
