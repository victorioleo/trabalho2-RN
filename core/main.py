from typing import List, Optional, Dict
from core.prompt import (base_prompt, end_prompt)
from core.utils import create_input_file_from_path, create_input_text
from data.vectorstore_manager import load_vectorstore_retriever
from langchain_core.documents import Document
from openai import OpenAI
from dotenv import load_dotenv
from core.load import process_file
import transformers
import torch

load_dotenv()

LIB_NAME = "postgraduate_regulations"
LLM_MODEL = "gpt-5-mini"

def generate_answer(
        question: str,
        lib_name: str = LIB_NAME,
        k: int = 5,
        model: Optional[str] = LLM_MODEL,
        local: Optional[bool] = False) -> str:
    """
    Generate an answer based on the provided question and context.
    Args:
        question: The student's question regarding the Postgraduate Regulations.
    Returns:
        A string containing the answer.
    """
    sanitized_query = question.replace('\x00', '')
    retriever = load_vectorstore_retriever(lib_name=lib_name, k=k)
    docs = retriever.invoke(sanitized_query)
    sanitized_docs = []
    for doc in docs:
        if isinstance(doc, Document):
            cleaned_content = doc.page_content.replace('\x00', '')
            sanitized_docs.append(doc.copy(update={'page_content': cleaned_content}))
        elif isinstance(doc, str):
            cleaned_content = doc.replace('\x00', '')
            sanitized_docs.append(Document(page_content=cleaned_content))

    prompt = base_prompt()
    prompt += f"""
    {sanitized_docs}
    """
    prompt += end_prompt()

    prompt += f"""
    {sanitized_query}
    """

    if local is False:
        client = OpenAI()
        content = []

        content.append(create_input_text(prompt))
        model = model

        response = client.responses.create (
            model=model,
            input=prompt
        )

        return response.output_text

    else:
        return llama3(prompt)

def init(path: Optional[str] = None):
    if path:
        process_file(path=path, lib=LIB_NAME)


def llama3(prompt: str) -> str:
    model_id = "meta-llama/Meta-Llama-3-8B"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    message = [
        {"role": "system", "content": "You are a helpful and precise academic assistant for the Faculty of Computer Science (FACOM) at UFMS."},
        {"role": "user", "content": prompt}
    ]

    outputs = pipeline(message, max_new_tokens=512)

    return outputs[0]['generated_text']