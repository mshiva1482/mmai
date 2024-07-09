from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile
from PyPDF2 import PdfReader
from langchain.schema import Document
from dotenv import load_dotenv
import os
from io import BytesIO
from langchain_community.llms import Ollama
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
import shutil
import ollama
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
import boto3
from botocore.exceptions import ClientError
import logging
from langchain_community.document_loaders import UnstructuredEmailLoader
import tempfile

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = '''
        Given the following text extracted from meeting minutes, generate a list of action items, their associated dates (if any) and the person/entity associated with the action item in JSON format. Each action item should be an object with the following properties:
        - "action": The specific task or action to be taken
        - "date": The due date or relevant date for the action (if mentioned), formatted as YYYY-MM-DD. If no date is specified, use null.
        - "entity": The person/group associated with the action item. If no person is associated, use null.

        Present the results as a JSON array of these objects. Ensure that each action item is clear, concise, and actionable. Ignore general discussion points or decisions that don't require specific actions.

        Meeting minutes text:

        {context}

        Please provide the JSON output of action items based on this text. If there are no action items in the meeting minutes, then return "None".
    '''


app = FastAPI()

@app.get("/")
async def root():
    logger.info("Hello, world")
    return {"message": "Hello World"}

@app.get("/get-action-items")
async def upload(file: UploadFile = File(...)):
    # Check file type
    file_extension = file.filename.split(".")[-1].lower()

    # Load documents
    if file_extension == "pdf":
        pdf_content = await file.read()
        pdf_file = BytesIO(pdf_content)
        pdf_reader = PdfReader(pdf_file)
        documents = []
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            documents.append(Document(page_content=text, metadata={"page": page_num + 1}))
        # logger.info("PDF file")
        # logger.info(documents)
    elif file_extension == "eml":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".eml") as tmp_file:
            tmp_file.write(await file.read())
            tmp_file_path = tmp_file.name
        loader = UnstructuredEmailLoader(tmp_file_path)
        documents = loader.load()
        os.remove(tmp_file_path)
        # logger.info("EML file")
        # logger.info(documents)

    text = "\n\n".join(doc.page_content for doc in documents)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=text)

    output = ollama.generate(
        model="mistral",
        prompt=prompt
    )

    response = {
        "ActionItems": output['response']
    }

    return response
