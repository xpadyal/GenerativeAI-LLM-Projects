import os
import streamlit as st
from langchain_cohere.llms import Cohere
from langchain_community.document_loaders import CSVLoader
from langchain_cohere import CohereEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import time
from langchain.chains.question_answering import load_qa_chain
import streamlit.components.v1 as components

# Set environment variables
os.environ["COHERE_API_KEY"] = "KzzamuEHYNDNBc65wgnxklRcngZA1agQC8UxOVu6"
os.environ["PINECONE_API_KEY"] = "9957e36c-76fc-428c-a38d-e9dc9778ad56"

# Initialize Cohere and Pinecone
llm = Cohere()
loader = CSVLoader("netflix_title.csv")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
embeddings = CohereEmbeddings()
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Create or use existing Pinecone index
index_name = "rag-test"
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=4096,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)
docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)

chain = load_qa_chain(llm, chain_type="stuff")

def chatbot_response(query):
    docs = docsearch.similarity_search(query, k=3)
    answer = chain.run(input_documents=docs, question=query)
    return answer

# Streamlit interface
st.set_page_config(page_title="Movie Recommendation Chatbot", page_icon=":clapper:", layout="centered")
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), url('https://wallpapercave.com/wp/wp1917118.jpg') no-repeat center center fixed;
        background-size: cover;
        color: #e5e5e5;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .chat-container {
        width: 100%;
        max-width: 700px;
        margin: auto;
        background-color: rgba(20, 20, 20, 0.8);
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .user-message, .bot-response {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #e50914;
        color: white;
        text-align: right;
    }
    .bot-response {
        background-color: #e50914;
        color: white;
        text-align: left;
    }
    .stTextInput, .stButton {
        width: 100%;
    }
    .stTextInput>div>input {
        background-color: #333;
        color: #e5e5e5;
        border: 1px solid #e50914;
    }
    .stTextInput>label {
        color: white;
    }
    .stButton>button {
        background-color: #e50914;
        color: white;
        border: none;
        padding: 10px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #f40612;
    }
    h1 {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Movie Recommendation Chatbot :clapper:")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your message:")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    response = chatbot_response(user_input)
    st.session_state.messages.append({"role": "bot", "content": response})

for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            f"""
            <div class='bot-response'>{message['content']}</div>
            """,
            unsafe_allow_html=True
        )
