import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# ------------------ Utility Functions ------------------ #
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    the provided context, just say, "Answer is not available in the context." Don't provide a wrong answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.session_state.chat_history.insert(0, {
        "question": user_question,
        "answer": response["output_text"]
    })


# ------------------ UI and Streamlit Layout ------------------ #
st.set_page_config(page_title="📄 PDF Chatbot", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background-image: url('https://img.freepik.com/free-photo/vivid-blurred-colorful-background_58702-2655.jpg?semt=ais_hybrid&w=740');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }
        [data-testid="stSidebar"] {
            background-image: url('https://images.unsplash.com/photo-1554034483-04fda0d3507b?fm=jpg&q=60&w=3000');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }
        [data-testid="stHeader"] {
            background-image: url('https://img.freepik.com/free-vector/dark-gradient-background-with-copy-space_53876-99548.jpg');
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            color: white;
            height: 80px;
        }
        .chat-entry {
            margin-bottom: 1.5rem;
            padding: 1rem;
            border-radius: 12px;
        }
        .user {
            background-color: #e0f7fa;
        }
        .bot {
            background-color: #fff3e0;
        }
    </style>
""", unsafe_allow_html=True)

st.title("💬 Chat with Your PDF (Gemini)")
st.caption("Powered by Gemini + LangChain + Streamlit")

with st.sidebar:
    st.title("📂 Upload and Process PDFs")
    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("✅ Done processing!")

with st.form("chat_form", clear_on_submit=True):
    user_question = st.text_input("Ask something about your PDFs")
    submitted = st.form_submit_button("Send")

if submitted and user_question:
    user_input(user_question)

# Display Chat History
for entry in st.session_state.chat_history:
    st.markdown(f"<div class='chat-entry user'>🙋‍♂️ <strong>You:</strong> {entry['question']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-entry bot'>🤖 <strong>AI:</strong> {entry['answer']}</div>", unsafe_allow_html=True)
