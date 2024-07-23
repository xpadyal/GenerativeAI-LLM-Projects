import os
import streamlit as st
from langchain_community.llms import EdenAI
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings.edenai import EdenAiEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import time
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import streamlit.components.v1 as components
import pandas as pd


# Set environment variables

os.environ["PINECONE_API_KEY"] = "9957e36c-76fc-428c-a38d-e9dc9778ad56"
EDENAI_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiMmI3NzU3ZWQtNmM2OC00MGZiLTk4Y2ItNzY4NWQ1MDA2YjE5IiwidHlwZSI6ImFwaV90b2tlbiJ9.OZmb2bq5MXIKZC9_BUXpaMrHyu3PjJ53nMrGlN7NV_s"
llm = EdenAI(edenai_api_key=EDENAI_API_KEY, provider="openai", temperature=0, max_tokens=250)

df = pd.read_csv("netflix_title.csv")

# Remove the 'show_id' column
df.drop(columns=['show_id','director','country','date_added'], inplace=True)

# Convert the remaining columns to a text file
with open("movie_postings.txt", "w") as f:
    for i, row in df.iterrows():
        heading = f"##Movie Detail: {i + 1}\n"
        sentence = f"The {row['type']} {row['title']} has cast of {row['cast']} and was released on {row['release_year']} with rating {row['rating']}.It has a duration of {row['duration']} in the genere of {row['listed_in']}.Here is a liitle description about the movie {row['description']}\n"
        f.write(heading)
        f.write(sentence)
        f.write("\n")

file_path = "movie_postings.txt"
print(f"File saved as {file_path}")

# Read the text file and split into chunks based on ##
with open(file_path, "r") as f:
    content = f.read()

textsplitter = [chunk for chunk in content.split("##") if chunk.strip()]


# Initialize Cohere and Pinecone
# loader = CSVLoader("netflix_title.csv")
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
# docs = text_splitter.split_documents(documents)
embeddings = EdenAiEmbeddings(edenai_api_key=EDENAI_API_KEY, provider="openai")
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Create or use existing Pinecone index
index_name = "rag-test"
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)
docsearch = PineconeVectorStore.from_texts(textsplitter, embeddings, index_name=index_name)

# chain = load_qa_chain(llm, chain_type="stuff")

def chatbot_response(query):
    knowledge = PineconeVectorStore.from_existing_index(
        index_name="rag-test",
        embedding=EdenAiEmbeddings(edenai_api_key=EDENAI_API_KEY, provider="openai")
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=knowledge.as_retriever(search_type="mmr", top_k=3)
    )
    
    prompt_template = PromptTemplate.from_template(
        "Act as a movie reccomedation assistant and recommend movies based on user query which is {userquery}. "
    )
    query = prompt_template.format(userquery= user_input)
    resultq = qa.invoke(query)
    
    return resultq['result']

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
