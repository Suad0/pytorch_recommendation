import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# Check if MPS is available and set the device accordingly
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Example tensor to verify MPS availability
if device.type == "mps":
    x = torch.ones(1, device=device)
    print(f"Tensor on MPS device: {x}")

# Load the dataset
df = pd.read_csv('/Users/suad/PycharmProjects/pytorch_recommendation/datasets/netflix_reviews_small.csv')
print(df.head())

# Load pre-trained SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Encode the review content
embeddings = model.encode(df['content'].tolist(), convert_to_tensor=True, device=device)

# Compute the cosine similarity matrix
cosine_scores = util.pytorch_cos_sim(embeddings, embeddings).to(device)


# Function to get top N similar reviews
# Function to get top N similar reviews
def get_recommendations(review_index, cosine_scores, df, top_n=3):
    scores = cosine_scores[review_index]
    top_results = torch.topk(scores, k=top_n + 1)  # +1 to skip the review itself
    recommendations = []
    print("Top results:", top_results)
    for score, idx in zip(top_results[0], top_results[1]):
        idx = int(idx.item())  # Convert tensor index to integer
        if idx != review_index:  # Skip the review itself
            recommendations.append((df.iloc[idx]['userName'], df.iloc[idx]['content'], score.item()))
    return recommendations


# Example: Get recommendations for the first review
recommendations = get_recommendations(0, cosine_scores, df)
print("Recommendations for user 'Megan Joel':")
for rec in recommendations:
    print(f"User: {rec[0]}, Review: {rec[1]}, Similarity Score: {rec[2]:.4f}")

if not torch.backends.mps.is_available():
    print("MPS device not found.")
