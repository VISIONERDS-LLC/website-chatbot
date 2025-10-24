from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List, Dict
from collections import deque, defaultdict
import uuid
import os
import random
from datetime import datetime, timedelta
import threading
import time
import logging
from rag_class_faiss import query_rag, reload_index

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BEARER_TOKEN = os.getenv("API_BEARER_TOKEN")
MAX_HISTORY_LENGTH = 10
SESSION_TIMEOUT_HOURS = 24

# Initialize HTTPBearer security scheme for Swagger
security = HTTPBearer()

# Welcome Messages and FAQs Data
WELCOME_MESSAGES = [
    "Hello! I'm here to help you learn about our IT services company and the innovative projects we've delivered. What would you like to know?",
    "Welcome! I can share insights about our technology stack, project experience, and service capabilities. How can I assist you today?",
    "Hi there! I'm your guide to understanding our company's expertise in software development, AI solutions, and digital transformation. What interests you?",
    "Greetings! I'm here to showcase our portfolio of successful projects and technical capabilities. What would you like to explore?",
    "Welcome to our company assistant! I can tell you about our cutting-edge projects, from AI-powered applications to scalable cloud solutions. What can I help with?",
    "Hello! Ready to discover how our team delivers custom software solutions and innovative technology implementations? Ask me anything!",
    "Hi! I'm here to help you understand our expertise in full-stack development, AI integration, and enterprise solutions. What would you like to know?",
    "Welcome! I can share details about our project methodologies, technology choices, and successful client collaborations. How can I help?",
    "Hello there! I'm your gateway to learning about our company's technical achievements and service offerings. What questions do you have?",
    "Greetings! I can provide insights into our development processes, technology expertise, and project success stories. What interests you most?"
]

FAQS = [
    "What types of projects does your company specialize in?",
    "What technologies do you primarily work with?",
    "Do you build AI-powered solutions?",
    "Can you handle both frontend and backend development?",
    "What industries do you serve?",
    "Do you provide cloud deployment and DevOps services?",
    "How do you ensure project quality and delivery?",
    "Can you integrate with existing systems and third-party APIs?",
    "Do you offer ongoing maintenance and support?",
    "How do you approach custom software development projects?"
]

class SessionStore:
    def __init__(self):
        self.sessions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_HISTORY_LENGTH))
        self.last_accessed: Dict[str, datetime] = {}
        self._lock = threading.Lock()
        self._start_cleanup_thread()
    
    def get_history(self, session_id: str) -> List[Dict]:
        with self._lock:
            self.last_accessed[session_id] = datetime.now()
            return list(self.sessions[session_id])
    
    def add_interaction(self, session_id: str, user_msg: str, assistant_msg: str):
        with self._lock:
            self.sessions[session_id].append({
                "user": user_msg,
                "assistant": assistant_msg,
                "timestamp": datetime.now().isoformat()
            })
            self.last_accessed[session_id] = datetime.now()
            logger.info(f"Added interaction to session {session_id[:8]}... | Messages in session: {len(self.sessions[session_id])}")
    
    def clear_session(self, session_id: str):
        with self._lock:
            if session_id in self.sessions:
                msg_count = len(self.sessions[session_id])
                del self.sessions[session_id]
                logger.info(f"Cleared session {session_id[:8]}... | Removed {msg_count} messages")
            if session_id in self.last_accessed:
                del self.last_accessed[session_id]
    
    def _cleanup_old_sessions(self):
        while True:
            time.sleep(3600)
            with self._lock:
                now = datetime.now()
                expired = [
                    sid for sid, last_time in self.last_accessed.items()
                    if now - last_time > timedelta(hours=SESSION_TIMEOUT_HOURS)
                ]
                if expired:
                    logger.info(f"Cleaning up {len(expired)} expired sessions")
                for sid in expired:
                    del self.sessions[sid]
                    del self.last_accessed[sid]
    
    def _start_cleanup_thread(self):
        thread = threading.Thread(target=self._cleanup_old_sessions, daemon=True)
        thread.start()
    
    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "total_sessions": len(self.sessions),
                "active_sessions": [
                    {
                        "session_id": sid[:16] + "...",
                        "messages": len(history),
                        "last_accessed": self.last_accessed[sid].isoformat()
                    }
                    for sid, history in self.sessions.items()
                ]
            }

# Initialize FastAPI
app = FastAPI(title="RAG Chatbot API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session_store = SessionStore()

@app.on_event("startup")
async def startup_event():
    """Load FAISS index on application startup"""
    logger.info("=" * 60)
    logger.info("Starting RAG Chatbot API v2.0.0")
    logger.info("Loading FAISS index...")
    success = reload_index()
    if success:
        logger.info("✓ FAISS index loaded successfully")
        logger.info("=" * 60)
    else:
        logger.error("✗ Failed to load FAISS index")
        raise RuntimeError("Failed to load FAISS index on startup")

# Updated verify_token function using HTTPBearer
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != BEARER_TOKEN:
        raise HTTPException(
            status_code=401, 
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Request/Response Models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[Dict]
    history_length: int

class WelcomeResponse(BaseModel):
    welcome_message: str
    faqs: List[str]

# Endpoints
@app.get("/")
async def root():
    return {
        "name": "RAG Chatbot API",
        "version": "2.0.0",
        "status": "running"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/welcome", response_model=WelcomeResponse)
async def get_welcome_and_faqs():
    """
    Get a random welcome message and 4 random FAQ questions for chatbot initialization
    """
    logger.info("Welcome endpoint called - generating random welcome message and FAQ questions")
    
    # Get 1 random welcome message
    welcome_message = random.choice(WELCOME_MESSAGES)
    
    # Get 4 random FAQ questions
    random_faqs = random.sample(FAQS, 4)
    
    logger.info(f"Returned welcome message: {welcome_message[:50]}...")
    logger.info(f"Returned {len(random_faqs)} random FAQ questions")
    
    return WelcomeResponse(
        welcome_message=welcome_message,
        faqs=random_faqs
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, token: str = Depends(verify_token)):
    session_id = request.session_id or str(uuid.uuid4())
    history = session_store.get_history(session_id)

    logger.info("=" * 60)
    logger.info(f"NEW CHAT REQUEST | Session: {session_id[:8]}...")
    logger.info(f"User Query: {request.message}")
    logger.info(f"History Length: {len(history)} messages")

    try:
        logger.info("Querying RAG system...")
        result = query_rag(
            query=request.message,
            history=history,
            model="gpt-5-nano"
        )

        # Log retrieved sources from FAISS index
        logger.info(f"FAISS Index Results: Retrieved {len(result['sources'])} sources")
        for idx, source in enumerate(result['sources'], 1):
            logger.info(f"  Source {idx}: {source.get('file', 'Unknown')} | Score: {source.get('score', 'N/A')}")
            logger.info(f"    Content preview: {source.get('content', '')[:100]}...")

        # Log OpenAI response
        logger.info(f"OpenAI Response Length: {len(result['answer'])} characters")
        logger.info(f"OpenAI Answer: {result['answer']}")

        session_store.add_interaction(
            session_id,
            request.message,
            result["answer"]
        )

        logger.info(f"✓ Request completed successfully")
        logger.info("=" * 60)

        return ChatResponse(
            session_id=session_id,
            answer=result["answer"],
            sources=result["sources"],
            history_length=len(history) + 1
        )

    except Exception as e:
        logger.error(f"✗ Query failed: {str(e)}", exc_info=True)
        logger.info("=" * 60)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/session/clear")
async def clear_session(session_id: str, token: str = Depends(verify_token)):
    logger.info(f"Clear session request for: {session_id[:8]}...")
    history = session_store.get_history(session_id)
    session_store.clear_session(session_id)
    return {
        "message": f"Cleared {len(history)} messages",
        "session_id": session_id
    }

@app.get("/session/{session_id}")
async def get_session(session_id: str, token: str = Depends(verify_token)):
    logger.info(f"Get session request for: {session_id[:8]}...")
    history = session_store.get_history(session_id)
    logger.info(f"Session {session_id[:8]}... has {len(history)} messages")
    return {
        "session_id": session_id,
        "history": history,
        "message_count": len(history)
    }

@app.get("/session/stats")
async def session_stats(token: str = Depends(verify_token)):
    logger.info("Session stats requested")
    stats = session_store.get_stats()
    logger.info(f"Total sessions: {stats['total_sessions']}")
    return stats

@app.post("/admin/reload-index")
async def reload_faiss_index(token: str = Depends(verify_token)):
    logger.info("Manual FAISS index reload requested")
    success = reload_index()
    if success:
        logger.info("✓ FAISS index reloaded successfully bro")
        return {"message": "FAISS index reloaded successfully"}
    else:
        logger.error("Failed to reload FAISS index")
        raise HTTPException(status_code=500, detail="Failed to reload index")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
