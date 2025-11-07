import torch
from sentence_transformers import util

def compute_similarity(user_input, model, movie_embeddings, num_series):
    # Encode user query
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    
    # Cosine similarity
    cos_scores = util.cos_sim(user_embedding, movie_embeddings)[0]
    top_indices = torch.topk(cos_scores, k=num_series).indices
    return top_indices.tolist()