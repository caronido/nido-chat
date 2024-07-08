import os
import logging
import requests
import io
from PIL import Image, ImageEnhance, ImageFilter
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from PyPDF2 import PdfReader
import pytesseract
from pdf2image import convert_from_bytes, pdfinfo_from_bytes
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")

# Initialize Slack app
app = App(token=SLACK_BOT_TOKEN)
app.documents = []  # Initialize with an empty list

# Initialize LangChain components
cached_llm = OpenAI(api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings()


def process_document(file_content, file_name):
    text = ""
    if file_name.endswith('.pdf'):
        try:
            pdf_reader = PdfReader(file_content)
            for page_num, page in enumerate(pdf_reader.pages, start=1):
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
                else:
                    # If no text is found, try OCR
                    images = convert_from_bytes(file_content.getvalue())
                    for image_num, image in enumerate(images, start=1):
                        logging.debug(f"Processing image {image_num}")
                        try:
                            image = preprocess_image(image)
                            ocr_text = pytesseract.image_to_string(image, lang='eng')
                            text += ocr_text + "\n"
                        except pytesseract.TesseractError as e:
                            logging.error(f"OCR error on page {page_num}, image {image_num}: {e}")
                            continue  # Skip problematic images
                        if len(text) > 5000:  # Add a condition to break loop if text gets too large
                            logging.debug("Breaking loop after large text accumulation")
                            break
                    break  # Break out of the outer loop once images are processed
        except Exception as e:
            raise ValueError(f"Error processing PDF: {e}")
    elif file_name.endswith(('.png', '.jpg', '.jpeg')):
        text = extract_text_from_image(file_content)
    else:
        raise ValueError("Unsupported file type")

    if not text.strip():
        raise ValueError("No text found in the document")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    texts = text_splitter.split_text(text)

    documents = [Document(page_content=text, metadata={"source": file_name}) for text in texts]

    if not documents:
        raise ValueError("No documents created from the text")

    # Generate unique IDs for each document
    ids = [f"{file_name}_{i}" for i in range(len(documents))]

    vector_store = Chroma.from_documents(documents=documents, embedding=embedding, ids=ids)

    return vector_store, documents

def preprocess_image(image):
    # Convert to grayscale
    image = image.convert('L')
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    # Apply a median filter to remove noise
    image = image.filter(ImageFilter.MedianFilter())
    return image

# OCR extraction function for image-based documents
def extract_text_from_image(file_content):
    image = Image.open(file_content)
    text = pytesseract.image_to_string(image)
    return text

# Handle document upload
@app.event("file_shared")
def handle_file_share(event, say):
    global cached_llm
    logging.debug(f"Received file_shared event: {event}")

    file_id = event["file"]["id"]
    file_info = app.client.files_info(file=file_id)
    file_url = file_info["file"]["url_private"]
    file_name = file_info["file"]["name"]

    # Download the file content
    headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
    response = requests.get(file_url, headers=headers)
    file_content = io.BytesIO(response.content)

    # Clear the existing vector store and documents
    app.vector_store = None
    app.documents = []

    try:
        # Process the file and create vector store
        vector_store, documents = process_document(file_content, file_name)
        app.vector_store = vector_store  # Store the latest vector store
        app.documents = documents  # Store the latest documents

        # Reinitialize the cached_llm to ensure the cache is cleared
        cached_llm = OpenAI(api_key=OPENAI_API_KEY)

        if app.documents:
            say(f"Document processed and ready for questions.")
        else:
            raise ValueError("No documents were processed.")
    except ValueError as e:
        logging.error(f"Error processing document: {e}")
        say(f"Error processing document: {e}")

# Chat with the document
@app.event("message")
def handle_message(message, say):
    logging.debug(f"Received message event: {message}")

    # Check if a document has been processed
    if not app.documents:
        logging.debug("No document has been processed yet.")
        say("No document has been processed yet.")
        return

    # Check for specific commands or queries
    user_message = message['text'].strip()
    if user_message.lower().startswith("chat with document"):
        logging.debug(f"User message: {user_message}")
        query = user_message.replace("chat with document", "").strip()
        response = ask_document(query)
        logging.debug(f"AI response: {response}")
        say(response["answer"] if "answer" in response else "No answer found.")
    else:
        logging.debug("Ignoring non-relevant message.")
        return

def ask_document(query):
    logging.debug(f"Query: {query}")

    # Create a prompt template that includes the context of the documents
    raw_prompt = PromptTemplate.from_template(
        """You are a VC expert at a fund called Nido Ventures. 
        The user will upload documents and you will answer questions based on the documents. 
        If you don't know the answer, say it. 
        Answer the following question based on the provided context.
        """
    )

    # Initialize the QA chain with the specified language model and chain type for each query
    qa_chain = load_qa_chain(cached_llm, chain_type="map_reduce")

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_documents(app.documents)

    all_answers = []
    for chunk in chunks:
        inputs = {
            "input_documents": [chunk],
            "question": query,
        }
        result = qa_chain(inputs)
        answer = result.get("output_text", "")
        all_answers.append(answer)

    # Combine all answers into a single response, ensuring it's concise and non-repetitive
    combined_answer = combine_answers(all_answers)

    if not combined_answer:
        combined_answer = "No answer found."

    response_answer = {
        "answer": combined_answer,
        "sources": [{"source": doc.metadata.get("source", "N/A"), "page_content": doc.page_content} for doc in app.documents]
    }
    return response_answer

def combine_answers(answers):
    # Deduplicate and summarize answers
    unique_answers = list(set(answers))  # Remove duplicate answers
    combined_text = " ".join(unique_answers).strip()

    # Use summarization to create a single coherent response
    summary_chain = load_summarize_chain(cached_llm, chain_type="map_reduce")
    summarized_answer = summary_chain.run({"input_documents": [Document(page_content=combined_text)]})

    return summarized_answer
