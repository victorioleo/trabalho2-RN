from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma 
from core.embeddings import CustomHFEmbeddings
import os
import re
import unicodedata
from dotenv import load_dotenv

load_dotenv()

embedding = CustomHFEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)

BASE_PERSIST_DIR = "subject_persist"

def _load_input(input_path : str):
    loader = PyMuPDFLoader(file_path=input_path)
    document = loader.load()

    return document

def create_vectorstore(input_path : str, lib_name : str):
    sanitized_lib_name = sanitize_collection_name(lib_name)

    if not os.path.exists(f'{BASE_PERSIST_DIR}/chroma_persist/{sanitized_lib_name}'):
        os.makedirs(f'{BASE_PERSIST_DIR}/chroma_persist/{sanitized_lib_name}')

    document = _load_input(input_path=input_path)
    splitted_documents = text_splitter.split_documents(documents=document)
    chroma = Chroma.from_documents(
        documents=splitted_documents,
        embedding=embedding,
        collection_name=sanitized_lib_name,
        persist_directory=f'{BASE_PERSIST_DIR}/chroma_persist/{sanitized_lib_name}'
    )

def update_vectorstore(input_path : str, 
                       lib_name : str):
    sanitized_lib_name = sanitize_collection_name(lib_name)
    if not os.path.exists(f'{BASE_PERSIST_DIR}/chroma_persist/{sanitized_lib_name}'):
        print('Chroma database not found')
    
    else:
        vector_store = Chroma(collection_name=sanitized_lib_name,
                              persist_directory=f'{BASE_PERSIST_DIR}/chroma_persist/{sanitized_lib_name}',
                              embedding_function=embedding,
                             )
        document = _load_input(input_path=input_path)
        splitted_documents = text_splitter.split_documents(documents=document)
        vector_store.add_documents(documents=splitted_documents)
        #print('Document inserted successfully!')
    
def load_vectorstore_retriever(lib_name : str,
                               k : int = 3):
    sanitized_lib_name = sanitize_collection_name(lib_name)
    vector_store = Chroma(collection_name=sanitized_lib_name,
                              persist_directory=f'{BASE_PERSIST_DIR}/chroma_persist/{sanitized_lib_name}',
                              embedding_function=embedding,
                             )
    retriever = vector_store.as_retriever(k = k)
    return retriever

def load_vectorstore(lib_name : str):
    sanitized_lib_name = sanitize_collection_name(lib_name)
    vector_store = Chroma(collection_name=sanitized_lib_name,
                              persist_directory=f'{BASE_PERSIST_DIR}/chroma_persist/{sanitized_lib_name}',
                              embedding_function=embedding,
                             )
    return vector_store

def sanitize_collection_name(name: str) -> str:
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode()
    name = name.replace(' ', '_')
    name = re.sub(r'[^a-zA-Z0-9._-]', '', name)
    name = re.sub(r'^[^a-zA-Z0-9]+', '', name)
    name = re.sub(r'[^a-zA-Z0-9]+$', '', name)
    if len(name) < 3:
        name = name.ljust(3, '_')
    if len(name) > 512:
        name = name[:512]
    return name