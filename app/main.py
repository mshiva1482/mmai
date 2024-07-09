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

load_dotenv()

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
    return {"message": "Hello World"}

@app.get("/get-action-items")
async def upload(file: UploadFile = File(...)):
    pdf_content = await file.read()
    pdf_file = BytesIO(pdf_content)
    pdf_reader = PdfReader(pdf_file)
    documents = []
    for page_num, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        documents.append(Document(page_content=text, metadata={"page": page_num + 1}))
    
    text = "\n\n".join(doc.page_content for doc in documents)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=text)

    logging.info(text)

    # output = ollama.generate(
    #     model="llama3",
    #     prompt=prompt
    # )

    return {
        # "ActionItems": output['response']
        "response": text
    }

