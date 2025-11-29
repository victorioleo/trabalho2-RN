from typing import List, Optional, Dict
from core.prompt import (base_prompt, end_prompt)
from core.utils import create_input_text
from data.vectorstore_manager import load_vectorstore_retriever
from langchain_core.documents import Document
from openai import OpenAI
from dotenv import load_dotenv
from core.load import process_file
import transformers
import torch
from core.factscore import FactScoreEvaluator

load_dotenv()

LIB_NAME = "postgraduate_regulations"
LLM_MODEL = "gpt-5-mini"

def main(
        question: str,
        lib_name: Optional[str] = LIB_NAME,
        k: int = 5,
        model: Optional[str] = LLM_MODEL,
        local: Optional[bool] = False) -> Dict:
    """
    Generate an answer based on the provided question and context.
    And calculate the factual accuracy score of the answer.
    Args:
        question: The student's question regarding the Postgraduate Regulations.
        lib_name: The name of the vectorstore library to use (default is LIB_NAME).
        k: The number of top documents chunks to retrieve (default is 5 chunks).
        model: The language model to use (default is LLM_MODEL).
        local: Whether to use a local model (Mistral) or the OpenAI API with the new Responses API.
    Returns:
        A dict containing the answer and factual accuracy score details.
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

    context = "\n".join([doc.page_content for doc in sanitized_docs])
    response = generate_answer(
        question=sanitized_query,
        context=context,
        model=model,
        local=local
    )

    #print(f"Generated Context: {sanitized_docs}\n")
    #print(f"Generated Answer: {response}")

    evaluator = FactScoreEvaluator(local=local, model=model)
    evaluation_score = evaluator.calculate_score(response, context)

    return {
        "answer": response,
        "total_facts": evaluation_score['total_facts'],
        "supported_facts": evaluation_score['supported_facts'],
        "unsupported_facts": evaluation_score['unsupported_facts'],
        "factual_accuracy_score": evaluation_score['factual_accuracy_score']
    }

def init(path: Optional[str] = None):
    if path:
        process_file(path=path, lib=LIB_NAME)


def local_model(prompt: str) -> str:
    try:
        model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "low_cpu_mem_usage": True,
            },
            device_map="auto",
        )
        
        formatted_prompt = f"[INST] You are a helpful academic assistant for FACOM at UFMS.\n\n{prompt} [/INST]"
        
        outputs = pipeline(
            formatted_prompt,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            return_full_text=False,
        )
        
        generated_text = outputs[0]['generated_text']
        return generated_text.strip()
        
    except Exception as e:
        return f"Error generating response locally: {str(e)}\nConsider using the OpenAI API (local=False)."
    

def generate_answer(
        question: str,
        context: str,
        model: Optional[str] = LLM_MODEL,
        local: Optional[bool] = False) -> str:
    """
    Generate an answer based on the provided question and context.
    Args:
        question: The student's question regarding the Postgraduate Regulations.
        context: The retrieved context to base the answer on.
        model: The language model to use (default is LLM_MODEL).
        local: Whether to use a local model (Mistral) or the OpenAI API with the new Responses API.
    Returns:
        A string containing the answer.
    """

    prompt = base_prompt()
    prompt += f"""
    {context}
    """
    prompt += end_prompt()

    prompt += f"""
    {question}
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
        return local_model(prompt)