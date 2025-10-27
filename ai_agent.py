import google.generativeai as genai
from typing import Dict, Any, List
from datetime import datetime
from config import settings
from database import User, Ticket
from sqlalchemy.orm import Session

class AIAgent:
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            tools=self._get_tools()
        )
    
    def _get_tools(self) -> List[Dict[str, Any]]:
        """Define the tools available to the AI agent"""
        return [
            {
                "function_declarations": [
                    {
                        "name": "create_ticket",
                        "description": "Creates a new support ticket for an IT issue",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "chatId": {"type": "string", "description": "User's chat ID"},
                                "problem": {"type": "string", "description": "Description of the IT problem"},
                                "solution": {"type": "string", "description": "Suggested solution for the problem"}
                            },
                            "required": ["chatId", "problem", "solution"]
                        }
                    },
                    {
                        "name": "get_past_tickets",
                        "description": "Fetches past support tickets for a user",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "chatId": {"type": "string", "description": "User's chat ID"}
                            },
                            "required": ["chatId"]
                        }
                    },
                    {
                        "name": "get_current_date",
                        "description": "Returns the current date and time"
                    }
                ]
            }
        ]
    
    async def create_ticket(self, chatId: str, problem: str, solution: str, db: Session) -> Dict[str, Any]:
        """Create a new support ticket"""
        try:
            # Ensure user exists
            user = db.query(User).filter(User.user_id == chatId).first()
            if not user:
                return {"success": False, "error": "User not found"}
            
            # Create ticket
            ticket = Ticket(
                user_id=chatId,
                problem_description=problem,
                solution_suggested=solution
            )
            db.add(ticket)
            db.commit()
            db.refresh(ticket)
            
            ticket_id = f"TICKET-{str(ticket.ticket_id).zfill(4)}"
            return {
                "success": True,
                "ticketId": ticket_id,
                "message": f"Ticket created: {ticket_id}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_past_tickets(self, chatId: str, db: Session) -> Dict[str, Any]:
        """Get past tickets for a user"""
        try:
            tickets = db.query(Ticket).filter(
                Ticket.user_id == chatId
            ).order_by(Ticket.created_at.desc()).limit(5).all()
            
            ticket_list = []
            for ticket in tickets:
                ticket_list.append({
                    "ticket_id": f"TICKET-{str(ticket.ticket_id).zfill(4)}",
                    "problem": ticket.problem_description,
                    "solution": ticket.solution_suggested,
                    "created_at": ticket.created_at.isoformat(),
                    "status": ticket.status
                })
            
            return {"success": True, "tickets": ticket_list}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_current_date(self) -> Dict[str, Any]:
        """Get current date and time"""
        now = datetime.now()
        return {
            "success": True,
            "result": {
                "day": now.day,
                "month": now.month,
                "year": now.year,
                "formatted": now.strftime("%d/%m/%Y"),
                "timestamp": now.isoformat()
            }
        }
    
    async def process_query(self, query: str, chatId: str, name: str, history: List[Dict], db: Session) -> str:
        """Process user query and return AI response"""
        try:
            # Create system instruction
            system_instruction = f"""
            You are an IT support chatbot for {name} (chatId: {chatId}).
            
            Guidelines:
            - Only handle queries about IT issues (laptop, Wi-Fi, software, hardware, network, etc.)
            - For off-topic queries, respond: "Sorry, I only handle IT support issues like laptop or Wi-Fi problems."
            - Never suggest harmful actions (hacking, unsafe hardware mods, data deletion without backups)
            - If unsure about safety, say: "Please consult a professional technician."
            - Avoid processing sensitive data (passwords, addresses, credit card info)
            - Be polite, concise, and professional
            - Address the user as {name}
            
            Behavior:
            - For greetings like "Hi", call get_past_tickets to summarize past issues or ask: "Hi {name}, how can I help with your IT issue today?"
            - For IT problems, provide a safe solution, then call create_ticket with chatId, problem summary, and solution
            - If unclear, ask for clarification without creating a ticket
            - Use conversation history for context: {history}
            
            Available tools:
            - create_ticket: Creates a support ticket
            - get_past_tickets: Gets user's past tickets
            - get_current_date: Gets current date/time
            """
            
            # Prepare conversation history
            conversation_history = []
            for msg in history:
                role = "assistant" if msg.get("isBot") else "user"
                conversation_history.append({
                    "role": role,
                    "parts": [{"text": msg.get("message", "")}]
                })
            
            # Start chat with history
            chat = self.model.start_chat(history=conversation_history)
            
            # Send query
            response = chat.send_message(query)
            
            # Process function calls if any
            final_response = ""
            while response.function_calls:
                function_calls = response.function_calls
                function_responses = []
                
                for call in function_calls:
                    function_name = call.name
                    args = dict(call.args)
                    
                    if function_name == "create_ticket":
                        result = await self.create_ticket(
                            args.get("chatId", chatId),
                            args.get("problem", ""),
                            args.get("solution", ""),
                            db
                        )
                    elif function_name == "get_past_tickets":
                        result = await self.get_past_tickets(
                            args.get("chatId", chatId),
                            db
                        )
                    elif function_name == "get_current_date":
                        result = await self.get_current_date()
                    else:
                        result = {"success": False, "error": "Unknown function"}
                    
                    function_responses.append({
                        "function_response": {
                            "name": function_name,
                            "response": result
                        }
                    })
                    
                    if result.get("success") and "message" in result:
                        final_response = result["message"]
                    elif result.get("success") and "tickets" in result:
                        tickets = result["tickets"]
                        if tickets:
                            final_response = f"Here are your recent tickets:\n"
                            for ticket in tickets:
                                final_response += f"• {ticket['ticket_id']}: {ticket['problem']} - {ticket['solution']}\n"
                        else:
                            final_response = "You don't have any past tickets. How can I help with your IT issue today?"
                
                # Send function responses back to the model
                if function_responses:
                    response = chat.send_message(function_responses)
                else:
                    break
            
            # Get final text response
            if not final_response and response.text:
                final_response = response.text
            
            return final_response or "I apologize, but I couldn't process your request. Please try again."
            
        except Exception as e:
            return f"An error occurred while processing your request: {str(e)}"

# Global AI agent instance
ai_agent = AIAgent()
