AI-Based Interview Monitoring & Evaluation System 🤖
Intern: Purva Sanjay Hande

Domain: Python / Artificial Intelligence

Focus: Context Window Optimization & Memory Retrieval Engineering

📌 Project Overview
The AI-Based Interview Monitoring & Evaluation System is designed to solve the "short-term memory" limitation of LLMs. By implementing a Retrieval-Augmented Generation (RAG) pattern, this module acts as the system's Long-Term Memory. It allows the AI to recall specific details from earlier in the interview, ensuring candidate responses are evaluated for consistency, depth, and factual accuracy without hitting token limits.

🛠️ Technical Stack
Backend Framework: FastAPI

Vector Database: FAISS (Facebook AI Similarity Search)

NLP Model: Sentence-Transformers (all-MiniLM-L6-v2)

Data Processing: NumPy

Language: Python 3.10+

🚀 Core Features
Vector Embeddings: Transforms raw interview transcripts into high-dimensional numerical vectors for semantic understanding.

FAISS Similarity Search: Enables sub-millisecond retrieval of relevant past interactions from the database.

Context Window Optimization: Dynamically injects only the most relevant historical context into prompts to minimize latency and token costs.

REST API Integration: Fully containerized logic exposed via FastAPI for easy integration with frontend or monitoring dashboards.

📂 File Structure
Plaintext
Ai_Project2.0/
├── main.py              # FastAPI application and API routes
├── faiss_memory.py      # Core logic for embeddings and FAISS search
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
⚙️ Installation & Setup
1. Prerequisites
Ensure you have Python 3.10+ installed.

2. Clone the Repository
Bash
git clone https://github.com/purvahande777/Ai_Project2.0.git
cd Ai_Project2.0
3. Install Dependencies
Bash
pip install fastapi uvicorn faiss-cpu sentence-transformers numpy pydantic
4. Run the Application
Bash
uvicorn main:app --reload
🧪 Testing the API
Once the server is running, you can test the endpoints (storing interview snippets and retrieving them) via the interactive Swagger UI:
👉 http://127.0.0.1:8000/docs
