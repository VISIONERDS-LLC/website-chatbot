from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List, Dict
from collections import deque, defaultdict
import uuid
import os
from datetime import datetime, timedelta
import threading
import time
from rag_class_faiss import query_rag, reload_index

BEARER_TOKEN = os.getenv("API_BEARER_TOKEN")
MAX_HISTORY_LENGTH = 10
SESSION_TIMEOUT_HOURS = 24

# Initialize HTTPBearer security scheme for Swagger
security = HTTPBearer()

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
    
    def clear_session(self, session_id: str):
        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
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

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, token: str = Depends(verify_token)):
    session_id = request.session_id or str(uuid.uuid4())
    history = session_store.get_history(session_id)
    
    try:
        result = query_rag(
            query=request.message,
            history=history,
            model="gpt-5-nano"
        )
        
        session_store.add_interaction(
            session_id, 
            request.message, 
            result["answer"]
        )
        
        return ChatResponse(
            session_id=session_id,
            answer=result["answer"],
            sources=result["sources"],
            history_length=len(history) + 1
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/session/clear")
async def clear_session(session_id: str, token: str = Depends(verify_token)):
    history = session_store.get_history(session_id)
    session_store.clear_session(session_id)
    return {
        "message": f"Cleared {len(history)} messages",
        "session_id": session_id
    }

@app.get("/session/{session_id}")
async def get_session(session_id: str, token: str = Depends(verify_token)):
    history = session_store.get_history(session_id)
    return {    
        "session_id": session_id,
        "history": history,
        "message_count": len(history)
    }

@app.get("/session/stats")
async def session_stats(token: str = Depends(verify_token)):
    return session_store.get_stats()

@app.post("/admin/reload-index")
async def reload_faiss_index(token: str = Depends(verify_token)):
    success = reload_index()
    if success:
        return {"message": "FAISS index reloaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload index")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
