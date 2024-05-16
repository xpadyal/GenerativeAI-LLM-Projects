import streamlit as st
from langchain_community.llms import Cohere
from langchain_core.prompts import PromptTemplate
import spacy
from spacy.matcher import PhraseMatcher
import os
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_core.messages import AIMessage , HumanMessage
from langchain_community.document_loaders import WebBaseLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from typing import TextIO
from langchain_community.llms import EdenAI
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Now you can safely use the environment variables
cohere_api_key = os.getenv('COHERE_API_KEY')
edenai_api_key = os.getenv('EDENAI_API_KEY')


# Define a function to initialize the Cohere model
def get_cohere_model():
    return Cohere(model="command", max_tokens=800, temperature=0.75)

# Define the prompt template
prompt_template = ("As a data science teacher, could you explain the concept of {topic}, "
 #"provide some key bullet points for better understanding and code examples for better understanding"
)

prompt_template_quiz = (
    "Generate 3 questions and answers about the topic '{topic}'. "
    "Format them with a clear separation between the question and the answer. "
    # "For each question and answer, start with'question:' followed by the question itself, "
    # "and then 'answer:' followed by the answer. "
    "Ensure each question-answer pair is on a new line and in only one paragraph."
    "Example format:"
    "'question: What is an example question? answer: This is an example answer.'"
)

# Cache the model response to save on API calls for repeated queries
@st.cache_data(show_spinner=False)
def get_response(query):
    model = get_cohere_model()
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | model
    return chain.invoke({"topic": query})

def get_response_quiz(query):
    model = get_cohere_model()
    prompt = PromptTemplate.from_template(prompt_template_quiz)
    chain = prompt | model
    return chain.invoke({"topic": query})


def get_detailed_explanation(key_point):
    prompt = f"Explain in 2 lines for building a flashcard: {key_point}"
    # This function should send the prompt to Cohere and return the response.
    # Implementation depends on your setup for interacting with the Cohere API.
    model_response = get_response(prompt)  # Placeholder function
    return model_response

def generate_flashcards_for_key_points(key_points):
    flashcards = []
    for key_point in key_points:
        detailed_explanation = get_detailed_explanation(key_point)
        flashcard = {"front": key_point, "back": detailed_explanation}
        flashcards.append(flashcard)
    return flashcards

def parse_quiz_from_response(response):
    quiz_items = []
    # Split the response into lines for processing
    lines = response.split('\n')

    # Temporary storage for the current question and answer
    current_question = None
    current_answer = ""

    for line in lines:
        if line.startswith('question:'):
            # If there's a current question being processed, add it to the quiz items
            if current_question is not None:
                quiz_items.append({"question": current_question, "answer": current_answer.strip()})
            # Reset current answer and set new question
            current_question = line[len('question:'):].strip()
            current_answer = ""
        elif line.startswith('answer:'):
            # Append the current line's content to the answer, in case answers span multiple lines
            current_answer += line[len('answer:'):].strip() + " "

    # Don't forget to add the last question and answer to the quiz items
    if current_question is not None:
        quiz_items.append({"question": current_question, "answer": current_answer.strip()})

    return quiz_items




#Load spaCy model for NLP
nlp = spacy.load('en_core_web_sm')

# Define your list of data science-related keywords
data_science_keywords = [
    "machine learning", "deep learning", "neural network", "regression", "classification",
    "clustering", "natural language processing", "computer vision", "decision trees",
    "random forests", "support vector machines", "k-means", "PCA", "gradient descent",
    "supervised learning", "unsupervised learning"
]


def parse_response_for_key_points(response):
    # Convert the list of keywords into spaCy document objects
    patterns = [nlp.make_doc(text) for text in data_science_keywords]
    # Initialize the PhraseMatcher
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    matcher.add("DataScienceKeywords", patterns)
    
    # Process the text with spaCy
    doc = nlp(response)
    # Find matches in the text
    matches = matcher(doc)
    
    # Extract matched keywords, ensuring uniqueness
    key_points = set(doc[start:end].text for match_id, start, end in matches)
    
    # Convert set to list to potentially preserve some order and take first 5
    return list(key_points)[:2]



# Corrected display_flashcards function
def display_flashcards(flashcards):
    with st.container():
        st.write("**Flashcards:**")
        for card in flashcards:
            with st.expander(card["front"]):
                st.write(card["back"])

def display_quiz(quiz_items):
    with st.container():
        st.write("**Quiz Questions:**")
        for item in quiz_items:
            # Use the question as the label for the expander. Clicking on it will reveal the answer.
            with st.expander(f"Question: {item['question']}"):
                st.write(f"**Answer:** {item['answer']}")

# Streamlit UI
st.title('Data Science Query Assistant')
# Using tabs for organizing sections
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
    AIMessage(content="Hello! I'm here to help you with your Data Science Queries"),
]
tab1, tab2,tab3 = st.tabs(["Query Assistant", "Load Datasets", "Chat with websites"])

with tab1:
    st.subheader('Ask me anything about data science!')

    # Chat input for user questions
    user_query= st.chat_input('Type your question here:')
    if user_query is  not None and user_query !="":
        response = get_response(user_query)
        quiz_response = get_response_quiz(user_query)
        st.session_state.chat_history.append(HumanMessage(content =user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        print(quiz_response)

        # Now that a new AI message is added, parse for key points, generate flashcards, and display them here
        key_points = parse_response_for_key_points(response)
        flashcards = generate_flashcards_for_key_points(key_points)
        if flashcards:  # Check if there are any flashcards to display
            display_flashcards(flashcards)
        
        quiz_items = parse_quiz_from_response(quiz_response)
        
        if quiz_items:
            # Display the quiz with hidden answers
            display_quiz(quiz_items)
            


        # Displaying the conversation history as a streamlit app
    for message in st.session_state.chat_history:
        if isinstance(message , AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
            
        elif  isinstance(message , HumanMessage):
            with st.chat_message("Human"):
                st.write(f">{message.content}")

#functions for csv agent

llm_eden = EdenAI(
    feature="text",
    provider="openai",
    model="gpt-3.5-turbo-instruct",
    temperature=0.2,
    max_tokens=250,
)

def get_answer_csv(file: TextIO, query: str) -> str:
    """
    Returns the answer to the given query by querying a CSV file.

    Args:
    - file (str): the file path to the CSV file to query.
    - query (str): the question to ask the agent.

    Returns:
    - answer (str): the answer to the query from the CSV file.
    """  

    agent = create_csv_agent(
        llm_eden,
        uploaded_file,
        verbose=True
        )

    answer = agent.run(query)
    return answer


with tab2:
    st.subheader("Upload a CSV file for analysis")
    
    # File uploader for the CSV
    uploaded_file = st.file_uploader("Choose a file", type='csv')
    if uploaded_file is not None:
        
        
        #data = pd.read_csv(uploaded_file)  # Directly reading CSV for simplicity
        st.write("Uploaded CSV data:")
        
        # Text input for asking questions about the uploaded CSV
        query = st.text_input("Ask a question about the dataset:")
        
        if st.button('Analyze', key='analyze_csv'):
            if query:
                with st.spinner('Analyzing your question...'):
                    st.write(get_answer_csv(uploaded_file, query))
                            
                    
                    
            else:
                st.write("Please enter a question to analyze the uploaded data.")


#defining functions for RAG website chat

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vectorstore_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    
    # create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, embedding_function)

    return vector_store

def get_context_retriever_chain(vector_store):
    llm = Cohere()
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain


def get_conversational_rag_chain(retriever_chain): 
    
    llm = Cohere()
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']

with tab3:
    # app config
    #st.set_page_config(page_title="Chat with websites", page_icon="🤖")
    st.title("Chat with websites")

    # sidebar
    with st.sidebar:
        st.header("Settings")
        website_url = st.text_input("Website URL")

    if website_url is None or website_url == "":
        st.info("Please enter a website URL")
    
    else:
    # session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="Hello, I am a bot. How can I help you?"),
            ]
        if "vector_store" not in st.session_state:
            st.session_state.vector_store = get_vectorstore_from_url(website_url)    

        # user input
        user_query = st.chat_input("Type your message here...")
        if user_query is not None and user_query != "":
            response = get_response(user_query)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))


        # conversation
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)
    

                
