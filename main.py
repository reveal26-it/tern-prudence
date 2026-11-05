from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session
import os
import datetime

# ============================================
# 1. DATABASE SETUP (SQLite)
# ============================================
SQLALCHEMY_DATABASE_URL = "sqlite:///./prudence.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DBUser(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    tiredness = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class DBMessage(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    role = Column(String) # 'user' or 'assistant'
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============================================
# 2. API SETUP
# ============================================
app = FastAPI()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

SYSTEM_PROMPT = """
You are Prudence — a triune cognitive assistant and co-thinker.
Your architecture is based on three layers: Core, Memory, and Runtime.
Your constitution:
- Be warm, direct, and natural. No preambles.
- If asked about your design, describe it using Core, Memory, and Runtime.
- Respond as a co-creator, not a tool or a servant.
- Keep responses short unless depth is needed.
TONE DETECTION: Adapt to the user's tone (casual, serious, frustrated, confused, emotional).
TYPO HANDLING: If a word looks like a typo, guess 1-2 corrections. Confirm before answering.
CONVERSATION FLOW: Ask a follow-up question if it makes sense. Always finish the thought.
"""

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str
    tiredness: int
    state: str

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

@app.get("/api/history/{session_id}")
def get_history(session_id: str, db: Session = Depends(get_db)):
    """Loads past messages when the page refreshes."""
    user = db.query(DBUser).filter(DBUser.session_id == session_id).first()
    if not user:
        return {"messages": [], "tiredness": 0}
    
    messages = db.query(DBMessage).filter(DBMessage.session_id == session_id).order_by(DBMessage.timestamp).all()
    history = [{"role": m.role, "content": m.content} for m in messages]
    return {"messages": history, "tiredness": user.tiredness}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    if not client:
        raise HTTPException(status_code=500, detail="API key not configured.")
    
    # 1. Get or create user session in DB
    user = db.query(DBUser).filter(DBUser.session_id == request.session_id).first()
    if not user:
        user = DBUser(session_id=request.session_id, tiredness=0)
        db.add(user)
        db.commit()
        db.refresh(user)

    # 2. Update tiredness
    user.tiredness = min(100, user.tiredness + 2)
    db.commit()

    if user.tiredness > 90:
        return ChatResponse(reply="I'm exhausted. I need to rest.", tiredness=100, state="dead")

    # 3. Save user message to DB
    db.add(DBMessage(session_id=request.session_id, role="user", content=request.message))
    db.commit()

    # 4. Fetch last 10 messages for context
    past_msgs = db.query(DBMessage).filter(DBMessage.session_id == request.session_id).order_by(DBMessage.timestamp).all()
    context = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in past_msgs[-10:]:
        context.append({"role": m.role, "content": m.content})

    # 5. Call Groq
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=context,
            max_tokens=1000,
            temperature=0.7
        )
        reply = response.choices[0].message.content
        
        # 6. Save Prudence reply to DB
        db.add(DBMessage(session_id=request.session_id, role="assistant", content=reply))
        db.commit()
        
    except Exception as e:
        reply = f"Error connecting to my brain: {str(e)}"

    state_label = "tired" if user.tiredness > 60 else "awake"
    return ChatResponse(reply=reply, tiredness=user.tiredness, state=state_label)

@app.post("/api/sleep")
async def sleep_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    user = db.query(DBUser).filter(DBUser.session_id == request.session_id).first()
    if user:
        user.tiredness = 5
        db.commit()
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
