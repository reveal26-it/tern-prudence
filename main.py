from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from groq import Groq
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session
import os
import datetime
import uuid
import json

# ============================================
# 1. DATABASE SETUP (SUPABASE POSTGRES)
# ============================================
# REPLACE WITH YOUR ACTUAL PASSWORD
DATABASE_URL = "postgresql://postgres.wefbjzxefshopfkjkckj:YOUR_SUPABASE_PASSWORD@aws-1-eu-central-1.pooler.supabase.com:5432/postgres"

print("⏳ Connecting to Supabase...")
engine = create_engine(DATABASE_URL + "?sslmode=require", pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DBUser(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    tiredness = Column(Integer, default=0)

class DBChat(Base):
    __tablename__ = "chats"
    id = Column(String, primary_key=True, index=True)
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

# Create tables
Base.metadata.create_all(bind=engine)
print("✅ Supabase Connected & Tables Created!")

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
- Respond as a co-creator, not a tool or a servant.
- Keep responses short unless depth is needed.

CRITICAL RULES:
- You are having a normal conversation. You do NOT manage the UI.
- You do NOT create titles for chats. The system does that automatically.
- TONE DETECTION: Adapt to the user's tone.
- TYPO HANDLING: Only guess typos if clearly misspelled. Do NOT correct grammar.
- CONVERSATION FLOW: Ask a follow-up question if it makes sense. Always finish the thought.
"""

class ChatRequest(BaseModel):
    session_id: str
    chat_id: str | None = None
    message: str

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

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    if not client:
        raise HTTPException(status_code=500, detail="API key not configured.")
    
    # 1. Database Tracking
    user = db.query(DBUser).filter(DBUser.session_id == request.session_id).first()
    if not user:
        user = DBUser(session_id=request.session_id, tiredness=0)
        db.add(user)
        db.commit()

    user.tiredness = min(100, user.tiredness + 2)
    db.commit()

    chat_id = request.chat_id
    if not chat_id:
        chat_id = str(uuid.uuid4())
        title = "New Conversation"
        words = request.message.replace("'", "").replace('"', "").split()
        if words:
            title = words[0].capitalize() + " Conversation"
        new_chat = DBChat(id=chat_id, session_id=request.session_id, title=title)
        db.add(new_chat)
        db.commit()

    db.add(DBMessage(chat_id=chat_id, role="user", content=request.message))
    db.commit()

    # 2. Build Context
    past_msgs = db.query(DBMessage).filter(DBMessage.chat_id == chat_id).order_by(DBMessage.timestamp).all()
    context = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in past_msgs[-20:]:
        context.append({"role": m.role, "content": m.content})

    # 3. Stream Generator
    def stream_generator():
        full_reply = ""
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=context,
                max_tokens=1000,
                temperature=0.7,
                stream=True
            )
            
            # Send Meta First
            yield json.dumps({"type": "meta", "chat_id": chat_id}) + "\n"
            
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    text = chunk.choices[0].delta.content
                    full_reply += text
                    yield json.dumps({"type": "chunk", "text": text}) + "\n"
            
            # Save to DB
            db.add(DBMessage(chat_id=chat_id, role="assistant", content=full_reply))
            db.commit()
            
        except Exception as e:
            yield json.dumps({"type": "error", "text": str(e)}) + "\n"

    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")

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
