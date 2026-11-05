from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session
import os
import datetime
import uuid

# ============================================
# 1. DATABASE SETUP
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

class DBChat(Base):
    __tablename__ = "chats"
    id = Column(String, primary_key=True, index=True) # UUID
    session_id = Column(String, index=True)
    title = Column(String, default="New Conversation")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class DBMessage(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(String, index=True)
    role = Column(String)
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

CRITICAL RULES:
- You are having a normal conversation. You do NOT manage the UI.
- You do NOT create titles for chats. The system does that automatically.
- You do NOT rename conversations. 
- If the user says "Remember the previous chat", they are just talking to you. Look at the conversation history provided in your context and respond to them naturally.

TONE DETECTION: Adapt to the user's tone.
TYPO HANDLING: If a word looks like a typo, guess 1-2 corrections. Confirm before answering.
CONVERSATION FLOW: Ask a follow-up question if it makes sense. Always finish the thought.
"""

class ChatRequest(BaseModel):
    session_id: str
    chat_id: str | None = None
    message: str

class ChatResponse(BaseModel):
    reply: str
    tiredness: int
    state: str
    chat_id: str

class ChatUpdate(BaseModel):
    title: str

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

@app.get("/api/chats/{session_id}")
def get_chat_list(session_id: str, db: Session = Depends(get_db)):
    chats = db.query(DBChat).filter(DBChat.session_id == session_id).order_by(DBChat.created_at.desc()).all()
    return [{"id": c.id, "title": c.title} for c in chats]

@app.get("/api/chat/{chat_id}")
def get_chat_history(chat_id: str, db: Session = Depends(get_db)):
    messages = db.query(DBMessage).filter(DBMessage.chat_id == chat_id).order_by(DBMessage.timestamp).all()
    return [{"role": m.role, "content": m.content} for m in messages]

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    if not client:
        raise HTTPException(status_code=500, detail="API key not configured.")
    
    # 1. Get or create User
    user = db.query(DBUser).filter(DBUser.session_id == request.session_id).first()
    if not user:
        user = DBUser(session_id=request.session_id, tiredness=0)
        db.add(user)
        db.commit()

    # 2. Update tiredness
    user.tiredness = min(100, user.tiredness + 2)
    db.commit()

    # 3. Get or create Chat with smart title
    chat_id = request.chat_id
    if not chat_id:
        chat_id = str(uuid.uuid4())
        
        stop_words = ["the", "this", "that", "chat", "title", "name", "call", "a", "an", "is", "can", "you", "to", "it", "let's"]
        words = request.message.replace("'", "").replace('"', "").split()
        notable_words = [w for w in words if w.lower() not in stop_words]
        
        if notable_words:
            title = notable_words[0].capitalize() + " Conversation"
        else:
            title = "New Conversation"
            
        new_chat = DBChat(id=chat_id, session_id=request.session_id, title=title)
        db.add(new_chat)
        db.commit()

    # 4. Save user message
    db.add(DBMessage(chat_id=chat_id, role="user", content=request.message))
    db.commit()

    # 5. Build context
    past_msgs = db.query(DBMessage).filter(DBMessage.chat_id == chat_id).order_by(DBMessage.timestamp).all()
    context = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in past_msgs[-10:]:
        context.append({"role": m.role, "content": m.content})

    # 6. Call Groq
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=context,
            max_tokens=1000,
            temperature=0.7
        )
        reply = response.choices[0].message.content
        
        # 7. Save AI reply
        db.add(DBMessage(chat_id=chat_id, role="assistant", content=reply))
        db.commit()
    except Exception as e:
        reply = f"Error: {str(e)}"

    state_label = "tired" if user.tiredness > 60 else "awake"
    return ChatResponse(reply=reply, tiredness=user.tiredness, state=state_label, chat_id=chat_id)

@app.put("/api/chats/{chat_id}")
def update_chat_title(chat_id: str, update: ChatUpdate, db: Session = Depends(get_db)):
    chat = db.query(DBChat).filter(DBChat.id == chat_id).first()
    if chat:
        chat.title = update.title
        db.commit()
    return {"status": "success"}

@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str, db: Session = Depends(get_db)):
    db.query(DBMessage).filter(DBMessage.chat_id == chat_id).delete()
    db.query(DBChat).filter(DBChat.id == chat_id).delete()
    db.commit()
    return {"status": "success"}

@app.post("/api/sleep")
async def sleep_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    user = db.query(DBUser).filter(DBUser.session_id == request.session_id).first()
    if user:
        user.tiredness = 5
        db.commit()
    return {"status": "success"}
