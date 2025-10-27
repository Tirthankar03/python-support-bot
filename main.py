from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
# asyncio import removed - not needed for simple FastAPI
from contextlib import asynccontextmanager

from config import settings
from database import create_tables, get_db, User, Ticket
from redis_client import redis_client
from ai_agent import ai_agent
from guardrails_config import guardrails
# WhatsApp bot removed - keeping only FastAPI

# Pydantic models
class SupportRequest(BaseModel):
    query: str
    chatId: str
    name: str

class SupportResponse(BaseModel):
    response: str

class TicketResponse(BaseModel):
    ticket_id: str
    problem: str
    solution: str
    created_at: str
    status: str

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting IT Support Chatbot...")
    # Tables are created by setup_database.py script
    print("Database ready!")
    
    yield
    
    # Shutdown
    print("Shutting down IT Support Chatbot...")

# Create FastAPI app
app = FastAPI(
    title="IT Support Chatbot",
    description="AI-powered IT support chatbot with memory and ticket management",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "IT Support Chatbot API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "IT Support Chatbot"}

@app.post("/support", response_model=SupportResponse)
async def support_endpoint(
    request: SupportRequest,
    db: Session = Depends(get_db)
):
    """Main support endpoint for processing IT support queries"""
    try:
        # Validate required fields
        if not request.query or not request.chatId or not request.name:
            raise HTTPException(
                status_code=400,
                detail="Query, chatId, and name are required"
            )
        
        # Check input guardrails
        input_check = await guardrails.check_input(request.query)
        if not input_check.valid:
            return SupportResponse(response=input_check.message)
        
        # Ensure user exists in database
        user = db.query(User).filter(User.user_id == request.chatId).first()
        if not user:
            user = User(user_id=request.chatId, name=request.name)
            db.add(user)
            db.commit()
        
        # Get conversation history
        history = await redis_client.get_conversation_history(request.chatId)
        
        # Save user message
        await redis_client.save_message(request.chatId, request.query, False)
        
        # Process with AI agent
        response = await ai_agent.process_query(
            request.query, 
            request.chatId, 
            request.name, 
            history, 
            db
        )
        
        # Check output guardrails
        output_check = await guardrails.check_output(response)
        if not output_check.valid:
            response = output_check.message
        
        # Save bot response
        await redis_client.save_message(request.chatId, response, True)
        
        return SupportResponse(response=response)
        
    except Exception as e:
        print(f"Error in support endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/tickets/{chat_id}", response_model=List[TicketResponse])
async def get_user_tickets(chat_id: str, db: Session = Depends(get_db)):
    """Get all tickets for a specific user"""
    try:
        tickets = db.query(Ticket).filter(
            Ticket.user_id == chat_id
        ).order_by(Ticket.created_at.desc()).all()
        
        ticket_list = []
        for ticket in tickets:
            ticket_list.append(TicketResponse(
                ticket_id=f"TICKET-{str(ticket.ticket_id).zfill(4)}",
                problem=ticket.problem_description,
                solution=ticket.solution_suggested,
                created_at=ticket.created_at.isoformat(),
                status=ticket.status
            ))
        
        return ticket_list
        
    except Exception as e:
        print(f"Error getting tickets: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/history/{chat_id}")
async def get_conversation_history(chat_id: str):
    """Get conversation history for a user"""
    try:
        history = await redis_client.get_conversation_history(chat_id)
        return {"history": history}
    except Exception as e:
        print(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/history/{chat_id}")
async def clear_conversation_history(chat_id: str):
    """Clear conversation history for a user"""
    try:
        await redis_client.clear_history(chat_id)
        return {"message": "Conversation history cleared"}
    except Exception as e:
        print(f"Error clearing history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
