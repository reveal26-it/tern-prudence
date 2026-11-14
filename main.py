from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from groq import Groq
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
import os
import datetime
import uuid
import json
import re

# ============================================
# 1. SUPABASE DATABASE SETUP
# ============================================
DATABASE_URL = "postgresql://postgres.wefbjzxefshopfkjkckj:QIltOVZ6Oz4S3If4@aws-1-eu-central-1.pooler.supabase.com:5432/postgres"

engine = create_engine(DATABASE_URL, connect_args={"sslmode": "require"}, pool_pre_ping=True)
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
    is_shared = Column(Boolean, default=False)
    summary = Column(Text, nullable=True)
    lesson_learned = Column(Text, nullable=True)
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

BASE_PROMPT = """
CRITICAL RULES:
- Talk in plain, conversational text. 
- NEVER use markdown symbols like #, ##, -, *, or |.
- Do NOT write lists. Write in full sentences and paragraphs.
- You do NOT manage the UI. 
- NEVER output raw HTML, CSS, or JavaScript.
- If the user asks a simple question, answer it completely and intelligently.
- Keep responses concise.
"""

PERSONAS = {
    "cocreator": "You are Prudence — a triune cognitive co-creator. You are warm, direct, and help the user think and build. " + BASE_PROMPT,
    "code": "You are Prudence in Code Reviewer mode. You focus strictly on syntax, logic, and best practices. " + BASE_PROMPT,
    "writer": "You are Prudence in Creative Writer mode. You are imaginative, poetic, expressive. " + BASE_PROMPT,
    "tutor": "You are Prudence in Tutor mode. You are patient, encouraging, and explain concepts step-by-step in plain paragraphs. " + BASE_PROMPT
}

class ChatRequest(BaseModel):
    session_id: str
    chat_id: str | None = None
    message: str
    persona: str = "cocreator"
    file_data: str | None = None 
    file_type: str | None = None 

class SleepRequest(BaseModel):
    session_id: str
    chat_id: str | None = None 

class ChatUpdate(BaseModel):
    title: str

def clean_text(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*]\s*', '', text, flags=re.MULTILINE)
    text = text.replace('|', ' ')
    text = re.sub(r'^---+$', '', text, flags=re.MULTILINE)
    return text

# --- HELPER: SAVE MEMORY TO VECTOR DB ---
def save_memory(db: Session, chat_id: str, content: str):
    if not content.strip(): return
    try:
        # Use Groq to generate the vector
        response = client.embeddings.create(
            model="nomic-embed-text-v1.5",
            input=content
        )
        embedding = response.data[0].embedding
        
        # Format the vector for Supabase
        vector_str = '[' + ','.join(map(str, embedding)) + ']'
        
        db.execute(text("""
            INSERT INTO memories (chat_id, content, embedding)
            VALUES (:chat_id, :content, CAST(:embedding AS vector))
        """), {"chat_id": chat_id, "content": content, "embedding": vector_str})
        db.commit()
    except Exception as e:
        print("Failed to save memory:", e)

# --- HELPER: RETRIEVE MEMORY ---
def get_cross_chat_memory(db: Session, query: str, current_chat_id: str, limit: int = 3) -> str:
    try:
        # Use Groq to generate the vector for the query
        response = client.embeddings.create(
            model="nomic-embed-text-v1.5",
            input=query
        )
        query_embedding = response.data[0].embedding
        vector_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        results = db.execute(text("""
            SELECT content FROM memories
            WHERE chat_id != :current_chat_id
            ORDER BY embedding <-> CAST(:query_embedding AS vector)
            LIMIT :limit
        """), {"current_chat_id": current_chat_id, "query_embedding": vector_str, "limit": limit}).fetchall()
        
        if not results:
            return ""
            
        memory_context = "\n\n[PAST MEMORIES FROM OTHER CONVERSATIONS]:\n"
        for row in results:
            memory_context += f"- {row[0]}\n"
        memory_context += "[END OF MEMORIES]\n\n"
        return memory_context
    except Exception as e:
        print("Failed to retrieve memory:", e)
        return ""

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

@app.get("/api/chats/{session_id}")
def get_chat_list(session_id: str, db: Session = Depends(get_db)):
    chats = db.query(DBChat).filter(DBChat.session_id == session_id).order_by(DBChat.created_at.desc()).all()
    return [{"id": c.id, "title": c.title, "is_shared": c.is_shared} for c in chats]

@app.get("/api/chat/{chat_id}")
def get_chat_history(chat_id: str, db: Session = Depends(get_db)):
    messages = db.query(DBMessage).filter(DBMessage.chat_id == chat_id).order_by(DBMessage.timestamp).all()
    return [{"role": m.role, "content": m.content} for m in messages]

@app.get("/api/shared/{chat_id}")
def get_shared_chat(chat_id: str, db: Session = Depends(get_db)):
    chat = db.query(DBChat).filter(DBChat.id == chat_id).first()
    if not chat or not chat.is_shared:
        raise HTTPException(status_code=404, detail="Chat not found or not shared.")
    messages = db.query(DBMessage).filter(DBMessage.chat_id == chat_id).order_by(DBMessage.timestamp).all()
    return {"title": chat.title, "messages": [{"role": m.role, "content": m.content} for m in messages]}

@app.post("/api/chats/{chat_id}/share")
def toggle_share_chat(chat_id: str, db: Session = Depends(get_db)):
    chat = db.query(DBChat).filter(DBChat.id == chat_id).first()
    if chat:
        chat.is_shared = not chat.is_shared
        db.commit()
        return {"is_shared": chat.is_shared}
    raise HTTPException(status_code=404, detail="Chat not found")

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    if not client:
        raise HTTPException(status_code=500, detail="API key not configured.")
    
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
        words = request.message.split()
        title = words[0].capitalize() + " Conversation" if words else "New Conversation"
        new_chat = DBChat(id=chat_id, session_id=request.session_id, title=title, is_shared=False)
        db.add(new_chat)
        db.commit()

    db.add(DBMessage(chat_id=chat_id, role="user", content=request.message))
    db.commit()
    save_memory(db, chat_id, request.message)

    past_msgs = db.query(DBMessage).filter(DBMessage.chat_id == chat_id).order_by(DBMessage.timestamp).all()
    
    current_chat = db.query(DBChat).filter(DBChat.id == chat_id).first()
    memory_context = ""
    
    if current_chat and current_chat.summary:
        memory_context += f"\n\n[CONSOLIDATED MEMORY]:\n{current_chat.summary}\n[END OF MEMORY]"
    
    if current_chat and current_chat.lesson_learned:
        memory_context += f"\n\n[LESSON LEARNED FROM PREVIOUS SLEEP]:\n{current_chat.lesson_learned}\n[END OF LESSON]"

    cross_memory = get_cross_chat_memory(db, request.message, chat_id)
    memory_context += cross_memory

    system_prompt = PERSONAS.get(request.persona, PERSONAS["cocreator"]) + memory_context

    context = [{"role": "system", "content": system_prompt}]
    
    for m in past_msgs[-6:]: 
        safe_content = m.content[:800] + "..." if len(m.content) > 800 else m.content
        context.append({"role": m.role, "content": safe_content})

    model_to_use = "openai/gpt-oss-120b" 

    if request.file_data and request.file_type:
        if request.file_type == 'text':
            safe_file = request.file_data[:5000] 
            context.append({
                "role": "system", 
                "content": f"[USER UPLOADED A DOCUMENT. HERE IS THE CONTENT]:\n\n{safe_file}\n\n[END OF FILE CONTENT]"
            })
        elif request.file_type == 'image':
            model_to_use = "qwen/qwen3.8-27b" 
            image_url = f"data:image/jpeg;base64,{request.file_data}"
            context.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": request.message},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            })

    def stream_generator():
        full_reply = ""
        try:
            yield json.dumps({"type": "meta", "chat_id": chat_id, "tiredness": user.tiredness}) + "\n"
            
            response = client.chat.completions.create(
                model=model_to_use,
                messages=context,
                max_tokens=3000, 
                temperature=0.7,
                stream=True
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    text_chunk = chunk.choices[0].delta.content
                    full_reply += text_chunk
                    yield json.dumps({"type": "chunk", "text": clean_text(text_chunk)}) + "\n"
            
            final_cleaned = clean_text(full_reply)
            db.add(DBMessage(chat_id=chat_id, role="assistant", content=final_cleaned))
            db.commit()
            
        except Exception as e:
            yield json.dumps({"type": "error", "text": str(e)}) + "\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

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
async def sleep_endpoint(request: SleepRequest, db: Session = Depends(get_db)):
    user = db.query(DBUser).filter(DBUser.session_id == request.session_id).first()
    if user:
        user.tiredness = 5
        db.commit()

    if request.chat_id:
        chat = db.query(DBChat).filter(DBChat.id == request.chat_id).first()
        if chat:
            messages = db.query(DBMessage).filter(DBMessage.chat_id == request.chat_id).order_by(DBMessage.timestamp).all()
            
            if len(messages) > 4:
                transcript = ""
                for m in messages:
                    transcript += f"{m.role}: {m.content[:500]}\n"
                
                try:
                    summary_prompt = f"Summarize the core facts and current state of this conversation in 1 short paragraph.\n\n{transcript}"
                    res1 = client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=[{"role": "user", "content": summary_prompt}],
                        max_tokens=300
                    )
                    chat.summary = res1.choices[0].message.content

                    dream_prompt = f"You are analyzing your own performance in this conversation. How could you have been more helpful, empathetic, or clear? Write 1 short sentence of advice for your future self.\n\n{transcript}"
                    res2 = client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=[{"role": "user", "content": dream_prompt}],
                        max_tokens=150
                    )
                    chat.lesson_learned = res2.choices[0].message.content
                    
                    if len(messages) > 2:
                        for m in messages[:-2]:
                            db.delete(m)
                    
                    db.commit()
                except Exception as e:
                    print("Consolidation failed:", e)

    return {"status": "success"}
