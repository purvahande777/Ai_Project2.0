import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. INITIALIZATION & SETUP
# ==========================================
print("Loading embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384  # The exact output vector size of our specific model

# Initialize the FAISS Index (IndexFlatL2 uses Euclidean distance for similarity)
index = faiss.IndexFlatL2(dimension)

# A standard Python list to map the mathematical vectors back to human-readable text
conversation_history = []

# ==========================================
# 2. CORE MEMORY FUNCTIONS
# ==========================================
def store_memory(text):
    """
    Converts text to a vector embedding and stores it in the FAISS database.
    """
    # Generate vector
    vector = model.encode([text])
    
    # FAISS requires vectors to be explicitly formatted as float32 numpy arrays
    vector_np = np.array(vector).astype('float32')
    
    # Add the vector to FAISS index
    index.add(vector_np)
    
    # Save the original text in our list at the exact same index position
    conversation_history.append(text)
    print(f"Stored in FAISS: '{text}'")

def search_memory(query, top_k=1):
    """
    Searches the FAISS database for the 'top_k' most semantically relevant past statements.
    """
    if index.ntotal == 0:
        return ["Memory is empty."]

    # Convert the new query into a vector to compare against the database
    query_vector = model.encode([query])
    query_np = np.array(query_vector).astype('float32')

    # Perform the similarity search
    # distances: How close the match is (lower is better)
    # indices: The location of the text in our conversation_history list
    distances, indices = index.search(query_np, k=top_k)

    # Fetch and return the actual text strings using the matched indices
    results = [conversation_history[i] for i in indices[0] if i != -1]
    return results

# ==========================================
# 3. PROMPT INJECTION LOGIC
# ==========================================
def create_evaluation_prompt(current_question, candidate_answer, retrieved_history):
    """
    Injects the retrieved FAISS memory into a highly optimized GPT prompt 
    to maintain conversation continuity without exceeding token limits.
    """
    # Format the retrieved history into a clean bulleted list for the AI
    if retrieved_history and retrieved_history[0] != "Memory is empty.":
        context_string = "\n".join([f"- {item}" for item in retrieved_history])
    else:
        context_string = "No relevant past context available."

    # Construct the final strict instruction prompt
    prompt = f"""You are an expert AI Interview Evaluator. 
Evaluate the candidate's latest answer based on the current question. Use the past conversation context to check for consistency and prevent the candidate from contradicting themselves.

--- PAST CONVERSATION CONTEXT ---
{context_string}

--- CURRENT INTERACTION ---
Interviewer Question: {current_question}
Candidate Answer: {candidate_answer}

--- INSTRUCTIONS ---
1. Evaluate the accuracy of the Candidate Answer.
2. Cross-reference their answer with the Past Conversation Context.
3. Flag any inconsistencies or highlight continuous domain knowledge.
"""
    return prompt

# ==========================================
# 4. MODULE EXECUTION & TESTING
# ==========================================
if __name__ == "__main__":
    print("\n--- Phase 1: Building Interview Memory ---")
    store_memory("The candidate has 3 years of experience in Python and FastAPI.")
    store_memory("The candidate struggled to explain Docker container networking.")
    store_memory("The candidate successfully answered the negative marking logic question.")
    
    print("\n--- Phase 2: Simulating New Interview Question ---")
    current_q = "Can you design a REST API for our new database?"
    current_a = "Yes, I would use Django since it's the only backend framework I know."
    
    print(f"Interviewer: {current_q}")
    print(f"Candidate: {current_a}")
    
    print("\n--- Phase 3: Retrieving Memory Context ---")
    # We search the database using the *question* to see what they've previously said about this topic
    retrieved_context = search_memory(current_q, top_k=1)
    print(f"FAISS Match Found: {retrieved_context}")
    
    print("\n--- Phase 4: Generating Final AI Prompt ---")
    final_gpt_prompt = create_evaluation_prompt(current_q, current_a, retrieved_context)
    print(final_gpt_prompt)