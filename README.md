Chat with Your Data (PDFs) using RAG
A Streamlit-based chatbot application that allows users to interact with their PDF documents using Retrieval-Augmented Generation (RAG) powered by Ollama and LangChain.

Overview
This project enables users to upload PDF documents and ask questions related to the content of those documents. The application uses a conversational AI model (Ollama) to generate responses based on the text extracted from the PDFs. The text is split into chunks, embedded using Ollama embeddings, and stored in a vector database (Chroma) for efficient retrieval.

Features
PDF Text Extraction: Extract text from uploaded PDF documents.

Text Chunking: Split the extracted text into manageable chunks for processing.

Vector Database: Store text chunks in a vector database for efficient retrieval.

Conversational AI: Use Ollama and LangChain to generate responses to user queries.

Streamlit UI: A user-friendly interface for uploading PDFs and interacting with the chatbot.

Installation
Clone the repository:

bash
Copy
git clone https://github.com/yourusername/chat-with-pdfs.git
cd chat-with-pdfs
Install dependencies:

bash
Copy
pip install -r requirements.txt
Set up environment variables:
Create a .env file in the root directory and add the following:

plaintext
Copy
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.langchain.plus
LANGCHAIN_API_KEY=your_langchain_api_key
Run the application:

bash
Copy
streamlit run app.py
Usage
Upload PDFs: Use the sidebar to upload one or more PDF documents.

Ask Questions: Enter your question in the text input field and press Enter.

View Responses: The chatbot will display responses based on the content of the uploaded PDFs.

Code Structure
app.py: The main application file that handles the Streamlit UI, PDF processing, and conversational AI.

htmlTemplates.py: Contains HTML and CSS templates for styling the chat interface.

Dependencies
Streamlit: For building the web application.

PyPDF2: For extracting text from PDF files.

LangChain: For managing conversational AI and document retrieval.

Ollama: For text embeddings and language model interactions.

Chroma: For vector database management.

Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.

Create a new branch (git checkout -b feature/YourFeatureName).

Commit your changes (git commit -m 'Add some feature').

Push to the branch (git push origin feature/YourFeatureName).

Open a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
