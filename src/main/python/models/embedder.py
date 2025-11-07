from sentence_transformers import SentenceTransformer

def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')