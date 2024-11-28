import streamlit as st
from dotenv import load_dotenv  # For loading environment variables from a .env file
from PyPDF2 import PdfReader  # For reading and extracting text from PDF files

#from langchain.document_loaders import PyPDFLoader
#Postimage is used for uploading the image which is used as icon.

# Importing useful tools for text embedding, splitting, and vector database management
from langchain_community.embeddings import OllamaEmbeddings  
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate

# Importing tools for conversational AI and retrieval
from langchain_community.chat_models import ChatOllama
from langchain.schema import Document
from langchain.retrievers.multi_query import MultiQueryRetriever

# For managing environment variables and API settings
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_API_KEY"] = "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"

# Importing additional tools for memory and UI templates
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template


def get_pdf_text(pdf_docs):
    """
    Extracts text content from a list of uploaded PDF files.
    
    Args:
        pdf_docs (list): List of uploaded PDF files.

    Returns:
        str: Combined text extracted from all the PDF pages.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()  # Extract text from each page
    return text


def get_text_chunks(text):
    """
    Splits the extracted text into manageable chunks for processing.
    
    Args:
        text (str): The full text extracted from PDF files.

    Returns:
        list: A list of Document objects, each containing a text chunk.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Size of each text chunk
        chunk_overlap=200,  # Overlap between consecutive chunks
        length_function=len  # Function to measure chunk length
    )
    chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    return documents


def get_vectorstore(text_chunks):
    """
    Creates a vector store from the provided text chunks using the Chroma library.

    Args:
        text_chunks (list): List of Document objects containing text chunks.

    Returns:
        Chroma: A vector store instance for retrieving relevant documents.
    """
    vectorstore = Chroma.from_documents(
        documents=text_chunks,
        embedding=OllamaEmbeddings(model='nomic-embed-text', temperature=1, show_progress=True),
        collection_name='local-rag'
    )
    return vectorstore


def get_conversation_chain(vectorstore):
    """
    Sets up a conversational retrieval chain using a vector store and an LLM.

    Args:
        vectorstore (Chroma): Vector store for retrieving relevant documents.

    Returns:
        ConversationalRetrievalChain: Configured chain for handling conversations.
    """
    local_model = 'mistral'  # Model name for the language model
    llm = ChatOllama(model=local_model)

    # Custom prompt for generating alternative queries
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from a vector
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search.
        Provide these alternative questions separated by newlines. 
        Original question: {question}
        Only provide the query, do not do numbering at the start of the questions.
        """
    )
    
    # Setting up a retriever with the custom prompt
    retriever = MultiQueryRetriever.from_llm(
        vectorstore.as_retriever(search_kwargs={"k": 5}),  # Retrieves top 5 results
        prompt=QUERY_PROMPT,
        llm=llm
    )
    
    # Memory to retain conversation history
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    
    # Conversational chain combining LLM, retriever, and memory
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    """
    Handles user input, retrieves responses from the conversational chain, and updates the UI.

    Args:
        user_question (str): The question provided by the user.
    """
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Render chat history in the Streamlit app
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:  # User messages
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:  # Bot messages
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    """
    Main function to initialize the Streamlit app and handle user interaction.
    """
    load_dotenv()  # Load environment variables
    
    st.set_page_config(
        page_title="Chat with your DATA (PDFs) using RAG",
        page_icon=":parrot:"
    )
    st.write(css, unsafe_allow_html=True) # renders provided css, allows for raw HTML content
    st.title("💬 Chatbot powered by Ollama")
    st.caption("🚀 Designed by :green[Jyotirmoy Gupta @ IIT_BHU]")
    st.caption("🚫 Make sure that you already downloaded the OLLAMA software in your System")

    # Initialize session state variables
    ''' 
    check if conversation variable is in the session state, 
    if not, initialize conversation to none 
    used to store the conversation history
    '''
    if "conversation" not in st.session_state:   
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with your DATA (PDFs) using RAG :parrot:")
    user_question = st.text_input("Ask anything related to your Data:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Kindly Upload your PDFs below & click on 'Upload'", 
            accept_multiple_files=True
        )
        if st.button("Upload"):
            with st.spinner("Take a deep breath for a While.."):
                # Extract text from uploaded PDFs
                raw_text = get_pdf_text(pdf_docs)

                # Split text into chunks
                text_chunks = get_text_chunks(raw_text)

                # Create vector store for document retrieval
                vectorstore = get_vectorstore(text_chunks)

                # Initialize conversational chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()
