from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from langchain.embeddings.base import Embeddings

MODEL_NAME = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Check if GPU is available and use it, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def make_embedding(input):
    input = input.strip().replace("\n", " ")
    # return_tensors="pt" returns PyTorch tensors
    inputs = tokenizer(input, return_tensors="pt", truncation=True, padding=True, max_length=8192).to(device) 
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        embeddings = F.normalize(last_hidden_state[:, 0], p=2, dim=1)
        return embeddings[0].cpu().numpy().tolist()

def embedding_function(texts):
    if isinstance(texts, str):
        texts = [texts]
    return [make_embedding(text) for text in texts]

class CustomHFEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return embedding_function(texts)

    def embed_query(self, text):
        return make_embedding(text)