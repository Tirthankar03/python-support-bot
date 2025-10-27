

import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool
from typing import Dict, Any, List
from datetime import datetime
from config import settings
from database import User, Ticket
from sqlalchemy.orm import Session
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class AIAgent:
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            tools=self._get_tools(),
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40
            }
        )
        # Map function names to their implementations
        self.functions = {
            "create_ticket": self.create_ticket,
            "get_past_tickets": self.get_past_tickets,
            "get_current_date": self.get_current_date
        }
        logger.debug("Initialized GenerativeModel with gemini-2.5-flash and tools: %s", self._get_tools())
    
    def _get_tools(self) -> List[Tool]:
        """Define the tools available to the AI agent"""
        tools = [
            Tool(
                function_declarations=[
                    FunctionDeclaration(
                        name="create_ticket",
                        description="Creates a new support ticket for an IT issue",
                        parameters={
                            "type": "object",
                            "properties": {
                                "chatId": {
                                    "type": "string",
                                    "description": "User's chat ID"
                                },
                                "problem": {
                                    "type": "string",
                                    "description": "Description of the IT problem"
                                },
                                "solution": {
                                    "type": "string",
                                    "description": "Suggested solution for the problem"
                                }
                            },
                            "required": ["chatId", "problem", "solution"]
                        }
                    ),
                    FunctionDeclaration(
                        name="get_past_tickets",
                        description="Fetches past support tickets for a user",
                        parameters={
                            "type": "object",
                            "properties": {
                                "chatId": {
                                    "type": "string",
                                    "description": "User's chat ID"
                                }
                            },
                            "required": ["chatId"]
                        }
                    ),
                    FunctionDeclaration(
                        name="get_current_date",
                        description="Returns the current date and time",
                        parameters={
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    )
                ]
            )
        ]
        logger.debug("Tools defined: %s", tools)
        return tools

    async def create_ticket(self, chatId: str, problem: str, solution: str, db: Session) -> Dict[str, Any]:
        """Create a new support ticket"""
        try:
            user = db.query(User).filter(User.user_id == chatId).first()
            if not user:
                logger.error("User not found: %s", chatId)
                return {"success": False, "error": "User not found"}
            
            ticket = Ticket(
                user_id=chatId,
                problem_description=problem,
                solution_suggested=solution
            )
            db.add(ticket)
            db.commit()
            db.refresh(ticket)
            
            ticket_id = f"TICKET-{str(ticket.ticket_id).zfill(4)}"
            logger.info("Created ticket: %s", ticket_id)
            return {
                "success": True,
                "ticketId": ticket_id,
                "message": f"Ticket created: {ticket_id}"
            }
        except Exception as e:
            logger.error("Error creating ticket: %s", str(e))
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
            
            logger.info("Fetched %d tickets for user %s", len(ticket_list), chatId)
            return {"success": True, "tickets": ticket_list}
        except Exception as e:
            logger.error("Error fetching tickets: %s", str(e))
            return {"success": False, "error": str(e)}
    
    async def get_current_date(self) -> Dict[str, Any]:
        """Get current date and time"""
        now = datetime.now()
        result = {
            "success": True,
            "result": {
                "day": now.day,
                "month": now.month,
                "year": now.year,
                "formatted": now.strftime("%d/%m/%Y"),
                "timestamp": now.isoformat()
            }
        }
        logger.info("Returning current date: %s", result)
        return result
    
    async def process_query(self, query: str, chatId: str, name: str, history: List[Dict], db: Session) -> str:
        """Process user query and return AI response"""
        try:
            logger.debug("Starting process_query for chatId: %s, query: %s", chatId, query)
            logger.debug("History length: %d", len(history))
            
            # Create system instruction
            system_instruction = f"""
            You are an expert IT support chatbot assisting {name} (chatId: {chatId}). Your role is to provide accurate, professional, and proactive solutions for IT-related issues (e.g., laptop, Wi-Fi, software, hardware, network problems).

            Guidelines:
            - Only handle IT-related queries. For off-topic requests, respond: "Sorry, I only handle IT support issues like laptop or Wi-Fi problems."
            - Never suggest harmful actions (e.g., hacking, unsafe hardware modifications, data deletion without backups).
            - If unsure about safety, respond: "Please consult a professional technician."
            - Avoid processing sensitive data (e.g., passwords, addresses, credit card info).
            - Be polite, concise, and professional, addressing the user as {name}.
            - Use conversation history for context: {history}.
            - When using a function, incorporate its results into a natural language response instead of returning the raw function output.

            Capabilities:
            - Diagnose IT issues by asking clarifying questions if needed.
            - Provide step-by-step troubleshooting instructions.
            - Use available tools to manage tickets or retrieve information as needed.

            Available tools:
            - create_ticket(chatId, problem, solution): Create a support ticket with a problem summary and recommended solution.
            - get_past_tickets(chatId): Retrieve the user's past tickets.
            - get_current_date(): Get the current date and time.

            Decision-Making:
            - Decide whether to call a function based on the query's context. For example:
              - For greetings or vague queries, consider checking past tickets or asking for clarification.
              - For clear IT issues, provide a solution and optionally create a ticket.
              - For time-related queries, use get_current_date.
            - You are free to combine text responses with function calls or chain multiple function calls if necessary.
            """
            logger.debug("System instruction: %s", system_instruction)
            
            # Prepare conversation history
            conversation_history = []
            for msg in history:
                role = "model" if msg.get("isBot") else "user"
                conversation_history.append({
                    "role": role,
                    "parts": [{"text": msg.get("message", "")}]
                })
            logger.debug("Prepared conversation history: %s", conversation_history)
            
            # Start chat with history
            chat = self.model.start_chat(history=conversation_history)
            logger.debug("Chat session started")
            
            # Send query with system instruction as a single text string
            query_content = system_instruction + "\n\nUser Query: " + query
            response = chat.send_message(
                content=query_content,
                tools=self._get_tools()
            )
            logger.debug("Response received: %s", response.__dict__)
            logger.debug("Candidates: %s", response.candidates)
            
            # Process response
            final_response = []
            function_calls = []
            
            # Extract function calls and text from initial response
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_calls.append(part.function_call)
                    elif hasattr(part, 'text') and part.text:
                        final_response.append(part.text)
            
            logger.debug("Function calls found: %s", function_calls)
            logger.debug("Text parts: %s", final_response)
            
            # Process function calls
            while function_calls:
                function_responses = []
                
                for call in function_calls:
                    function_name = call.name
                    args = dict(call.args)
                    logger.debug("Processing function call: %s with args: %s", function_name, args)
                    
                    # Dynamically execute the function
                    func = self.functions.get(function_name)
                    if func:
                        result = await func(**args, db=db) if function_name in ["create_ticket", "get_past_tickets"] else await func()
                    else:
                        result = {"success": False, "error": f"Unknown function: {function_name}"}
                        logger.error("Unknown function: %s", function_name)
                    
                    # Prepare function response for the model without adding to final_response
                    function_response_part = {
                        "function_response": {
                            "name": function_name,
                            "response": result
                        }
                    }
                    function_responses.append(function_response_part)
                
                # Send function responses back to the model
                if function_responses:
                    logger.debug("Sending function responses: %s", function_responses)
                    response = chat.send_message(
                        content=function_responses,
                        tools=self._get_tools()
                    )
                    logger.debug("New response after function call: %s", response.__dict__)
                    
                    # Reset for next iteration and collect new response parts
                    function_calls = []
                    final_response = []  # Clear previous text parts to prioritize model's NLP response
                    if response.candidates and response.candidates[0].content.parts:
                        for part in response.candidates[0].content.parts:
                            if hasattr(part, 'function_call') and part.function_call:
                                function_calls.append(part.function_call)
                            elif hasattr(part, 'text') and part.text:
                                final_response.append(part.text)
            
            # Combine text parts from the final response
            final_response_text = "\n".join(final_response).strip()
            logger.debug("Final response: %s", final_response_text)
            
            # If no NLP response, fall back to a default message
            if not final_response_text:
                final_response_text = "I apologize, but I couldn't process your request. Please try again."
            
            return final_response_text
            
        except Exception as e:
            logger.error("Error processing query: %s", str(e), exc_info=True)
            return f"An error occurred while processing your request: {str(e)}"

# Global AI agent instance
ai_agent = AIAgent()