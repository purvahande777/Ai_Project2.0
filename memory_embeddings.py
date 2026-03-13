from sentence_transformers import SentenceTransformer

# 1. Load the open-source embedding model
# 'all-MiniLM-L6-v2' is fast, lightweight, and perfect for local testing
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embedding(text):
    """
    Converts a text string into a dense vector array.
    """
    # The encode method turns the string into numbers
    vector = model.encode(text)
    return vector

# --- Test the function ---
if __name__ == "__main__":
    sample_answer = "I have two years of experience building APIs with FastAPI."
    
    # Generate the embedding
    vector_result = generate_embedding(sample_answer)
    
    print(f"\nOriginal Text: '{sample_answer}'")
    print(f"Vector Dimensions: {vector_result.shape} (This means it created 384 numbers)")
    print(f"First 5 numbers of the vector: \n{vector_result[:5]}")