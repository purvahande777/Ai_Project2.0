from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. SETUP & INITIALIZATION
# ==========================================
app = FastAPI(title="Interview Memory & Context API")

print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384
index = faiss.IndexFlatL2(dimension)
conversation_history = []

# ==========================================
# 2. DATA MODELS (Defining what the API expects to receive)
# ==========================================
class MemoryItem(BaseModel):
    text: str

class EvaluationRequest(BaseModel):
    current_question: str
    candidate_answer: str
    top_k: int = 1

# ==========================================
# 3. API ENDPOINTS
# ==========================================
@app.post("/store_memory/")
def api_store_memory(item: MemoryItem):
    """Endpoint to save a new interaction into the FAISS database."""
    vector = model.encode([item.text])
    vector_np = np.array(vector).astype('float32')
    
    index.add(vector_np)
    conversation_history.append(item.text)
    
    return {"status": "success", "message": f"Successfully stored: '{item.text}'"}

@app.post("/generate_prompt/")
def api_generate_prompt(request: EvaluationRequest):
    """Endpoint to search memory and generate the optimized GPT prompt."""
    if index.ntotal == 0:
        retrieved_history = ["Memory is empty."]
    else:
        # Search logic
        query_vector = model.encode([request.current_question])
        query_np = np.array(query_vector).astype('float32')
        distances, indices = index.search(query_np, k=request.top_k)
        retrieved_history = [conversation_history[i] for i in indices[0] if i != -1]

    # Format the prompt
    if retrieved_history and retrieved_history[0] != "Memory is empty.":
        context_string = "\n".join([f"- {item}" for item in retrieved_history])
    else:
        context_string = "No relevant past context available."

    final_prompt = f"""You are an expert AI Interview Evaluator. 
Evaluate the candidate's latest answer based on the current question. Use the past conversation context to check for consistency.

--- PAST CONVERSATION CONTEXT ---
{context_string}

--- CURRENT INTERACTION ---
Interviewer Question: {request.current_question}
Candidate Answer: {request.candidate_answer}

--- INSTRUCTIONS ---
1. Evaluate the accuracy of the Candidate Answer.
2. Cross-reference their answer with the Past Conversation Context.
3. Flag any inconsistencies or highlight continuous domain knowledge.
"""
    
    return {
        "status": "success",
        "retrieved_context": retrieved_history,
        "optimized_gpt_prompt": final_prompt
    }