import pandas
import torch
from sentence_transformers import SentenceTransformer, util

df = pandas.read_csv('/Users/suad/PycharmProjects/pytorch_recommendation/datasets/netflix_reviews.csv')

print(df.head())

# Load pre-trained SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Encode the review content
embeddings = model.encode(df['content'].tolist(), convert_to_tensor=True)

# Compute the cosine similarity matrix
cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)


# Function to get top N similar reviews
def get_recommendations(review_index, cosine_scores, df, top_n=3):
    scores = cosine_scores[review_index]
    top_results = torch.topk(scores, k=top_n + 1)  # +1 to skip the review itself
    recommendations = []
    for score, idx in zip(top_results[0], top_results[1]):
        if idx != review_index:  # Skip the review itself
            recommendations.append((df.iloc[idx]['userName'], df.iloc[idx]['content'], score.item()))
    return recommendations


# Example: Get recommendations for the first review
recommendations = get_recommendations(0, cosine_scores, df)
print("Recommendations for user 'Megan Joel':")
for rec in recommendations:
    print(f"User: {rec[0]}, Review: {rec[1]}, Similarity Score: {rec[2]:.4f}")

# This code takes user reviews from a CSV file, encodes them into vector representations,
# computes similarities between all reviews, and then generates recommendations for a specified
# review based on those similarities. The recommendations are reviews that are most similar to the specified review.
# This can be useful for suggesting similar content or features to users based on their review content.
