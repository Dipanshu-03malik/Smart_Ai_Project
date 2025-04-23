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
#import openai
import requests
import time

# Load environment variables
load_dotenv()
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
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)
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
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    driver.get(url)
    time.sleep(5)
    page_source = driver.page_source
    driver.quit()
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
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        
        # Ensure the chat history is initialized
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        # Prepare chat history with the expected format
        chat_history = [
            {"role": "user", "parts": [{"text": message['content']}]}
            if message['role'] == "You" 
            else {"role": "model", "parts": [{"text": message['content']}]}
            for message in st.session_state['chat_history']
        ]

        # Start a new chat session with the correctly formatted history
        chat = model.start_chat(history=chat_history)

        def get_gemini_response(question):
            response = chat.send_message(question, stream=True)
            return response

        response = get_gemini_response(user_question)
        response_text = "".join(chunk.text for chunk in response)

        st.write("Reply: ", response_text)

        # Append new entries to the chat history in session state
        st.session_state['chat_history'].append({"role": "You", "content": user_question})
        st.session_state['chat_history'].append({"role": "Bot", "content": response_text})

        # Display chat history
        st.subheader("The Chat History is")
        for message in st.session_state['chat_history']:
            st.write(f"{message['role']}: {message['content']}")



# Function to generate an emoji sticker using the TraxDinosaur Emoji API
# Function to generate an emoji sticker using the TraxDinosaur Emoji API
def generate_emoji():
    st.markdown("<h2 style='color:#fc1008'>Generate Emoji Sticker</h2>", unsafe_allow_html=True)

    prompt = st.text_input("Enter a prompt for emoji generation")

    if prompt and st.button("Generate Emoji"):
        st.warning("Generating emoji, please wait...")
        
        # Define the API endpoint and headers
        api_url = "https://apiemojistrax.onrender.com/api/genemoji"  # Updated API URL
        headers = {
            "Content-Type": "application/json"
        }
        body = {
            "prompt": prompt
        }

        # Send POST request to the API
        response = requests.post(api_url, headers=headers, json=body)

        if response.status_code == 200:
            # If the response is successful, display the emoji sticker
            emoji_image = response.content
            st.image(emoji_image, caption="Generated Emoji Sticker", use_column_width=True)
        else:
            st.error("Error generating emoji: " + response.text)

# Function to genraate an uploaded image using AI
# Function to generate an uploaded image using AI
def generate_image():
    st.markdown("<h2 style='color:#fc1008'>Generate Image using TraxDinosaur API</h2>", unsafe_allow_html=True)

    prompt = st.text_input("Enter a prompt for image generation")

    if prompt and st.button("Generate Image"):
        st.warning("Generating image, please wait...")
        
        # Define the API endpoint and headers
        api_url = "https://apiimagestrax.vercel.app/api/genimage"
        headers = {
            "Content-Type": "application/json"
        }
        body = {
            "prompt": prompt
        }

        # Send POST request to the API
        response = requests.post(api_url, headers=headers, json=body)

        if response.status_code == 200:
            # If the response is successful, display the image
            st.image(response.content, caption="Generated Image", use_column_width=True)
        else:
            st.error("Error generating image: " + response.text)
# Main function
# Main function
def main():
    st.set_page_config("Chat Options", layout="wide")

    if 'sidebar_state' not in st.session_state:
        st.session_state['sidebar_state'] = 'collapsed'

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

    if st.button("☰ Smart Ai", key="toggle_sidebar"):
        if st.session_state['sidebar_state'] == 'collapsed':
            st.session_state['sidebar_state'] = 'expanded'
        else:
            st.session_state['sidebar_state'] = 'collapsed'

    if st.session_state['sidebar_state'] == 'expanded':
        with st.sidebar:
            st.markdown("<h1 style='color:red'>Menu</h1>", unsafe_allow_html=True)
            options = ["Chat with PDF", "Chat with Webpage", "Chat with AI", "Generate Image", "Generate Emoji"]
            selected_option = st.selectbox("Select an option", options)

            if selected_option == "Chat with PDF":
                chat_with_pdf()
            elif selected_option == "Chat with Webpage":
                chat_with_webpage()
            elif selected_option == "Chat with AI":
                chat_with_ai()
            elif selected_option == "Generate Image":
                generate_image()
            elif selected_option == "Generate Emoji":
                generate_emoji()

if __name__ == "__main__":
    main()