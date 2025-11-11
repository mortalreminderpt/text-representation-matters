import os
import pickle

from typing import Dict, List

import torch
import openai
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

BERT_CACHE_PATH = "bert_embedding_cache.pkl"
LLM_CACHE_PATH  = "openai_embedding_cache.pkl"

def load_cache(path: str) -> Dict[str, np.ndarray]:
    """
    Load cache from path
    
    :param path: path of cache file
    """
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}

def save_cache(cache: Dict[str, np.ndarray], path: str):
    """
    Save cache to path
    
    :param cache: cache dict
    :param path: path of cache file
    """
    with open(path, "wb") as f:
        pickle.dump(cache, f)

def init_bert():
    """
    Initialize BERT tokenizer and model
    """
    global bert_tokenizer, bert_model
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_model     = AutoModel.from_pretrained("bert-base-uncased").eval()

def bert_encode_sentences(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Encode sentences using BERT
    
    :param texts: text list
    :param batch_size: batch size
    :return: embeddings numpy array
    """
    cache = load_cache(BERT_CACHE_PATH)
    embeddings, new_texts = [], []
    for t in texts:
        if t in cache:
            embeddings.append(cache[t])
        else:
            new_texts.append(t)
    if new_texts:
        init_bert()
        for i in tqdm(range(0, len(new_texts), batch_size), desc="BERT-Embedding"):
            batch = new_texts[i:i+batch_size]
            tokens  = bert_tokenizer(batch, padding=True, truncation=True,
                                   return_tensors="pt", max_length=128)
            with torch.no_grad():
                out = bert_model(**tokens)
            vector = out.last_hidden_state[:,0,:].cpu().numpy()
            for text, embedding in zip(batch, vector):
                cache[text] = embedding
                embeddings.append(embedding)
        save_cache(cache, BERT_CACHE_PATH)
    else:
        embeddings = [cache[t] for t in texts]
    return np.vstack(embeddings)

def init_openai(api_key: str = None):
    """
    Initialize OpenAI
    
    :param api_key: OpenAI API key
    """
    if api_key:
        openai.api_key = api_key
    elif 'OPENAI_API_KEY' not in os.environ:
        raise ValueError("Set OPENAI_API_KEY env or pass api_key.")
    return openai

def openai_encode_sentences(texts: List[str],
                         model: str = "text-embedding-3-large",
                         batch_size: int = 256) -> np.ndarray:
    """
    Encode sentences using OpenAI
    
    :param texts: text list
    :param model: OpenAI model
    :param batch_size: batch size
    :return: embeddings numpy array
    """
    cache = load_cache(LLM_CACHE_PATH)
    embeddings, new_texts = [], []
    for t in texts:
        if t in cache:
            embeddings.append(cache[t])
        else:
            new_texts.append(t)
    if new_texts:
        openai = init_openai()
        for i in tqdm(range(0, len(new_texts), batch_size), desc="OpenAI-Embedding"):
            batch = new_texts[i:i+batch_size]
            response = openai.embeddings.create(input=batch, model=model)
            vector = [np.array(d.embedding) for d in response.data]
            for text, embedding in zip(batch, vector):
                cache[text] = embedding
                embeddings.append(embedding)
        save_cache(cache, LLM_CACHE_PATH)
    else:
        embeddings = [cache[t] for t in texts]
    return np.vstack(embeddings)