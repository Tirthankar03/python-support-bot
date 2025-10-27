from guardrails import Guard
from guardrails.hub import ProfanityFree, ToxicLanguage, NoPersonalInfo
from pydantic import BaseModel
from typing import Dict, Any

class GuardrailResponse(BaseModel):
    valid: bool
    message: str = ""

class ITSupportGuardrails:
    def __init__(self):
        # Initialize NVIDIA Guardrails with multiple safety checks
        self.guard = Guard().use(
            ProfanityFree(),
            ToxicLanguage(),
            NoPersonalInfo()
        )
        
        # IT-specific banned terms
        self.it_banned_terms = [
            "hack", "password", "credit card", "address", "ssn", "social security",
            "bank account", "pin", "cvv", "cvc", "expiry", "expiration"
        ]
        
        # Off-topic patterns
        self.off_topic_patterns = [
            r"weather", r"math", r"personal", r"politics", r"cooking", 
            r"travel", r"dating", r"relationship", r"medical", r"legal"
        ]
    
    async def check_input(self, text: str) -> GuardrailResponse:
        """Check if input text passes all guardrails"""
        try:
            # Check for IT-specific banned terms
            text_lower = text.lower()
            for term in self.it_banned_terms:
                if term in text_lower:
                    return GuardrailResponse(
                        valid=False,
                        message="I can't process requests involving sensitive or unsafe terms. Please rephrase your IT support request."
                    )
            
            # Check for off-topic content
            import re
            for pattern in self.off_topic_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return GuardrailResponse(
                        valid=False,
                        message="Sorry, I only handle IT support issues like laptop or Wi-Fi problems. Please describe your technical issue."
                    )
            
            # Use NVIDIA Guardrails for additional safety checks
            try:
                validated_text, validated_output, guard_history = self.guard.validate(
                    text, 
                    metadata={"prompt": "IT support query"}
                )
                
                # If guardrails pass, return success
                return GuardrailResponse(valid=True)
                
            except Exception as e:
                # If guardrails fail, return appropriate message
                return GuardrailResponse(
                    valid=False,
                    message="I can't process this request due to safety concerns. Please rephrase your IT support question."
                )
                
        except Exception as e:
            # Fallback to basic check if guardrails fail
            return GuardrailResponse(
                valid=False,
                message="I can't process this request. Please describe your IT support issue clearly."
            )
    
    async def check_output(self, text: str) -> GuardrailResponse:
        """Check if AI response is safe to send"""
        try:
            # Use NVIDIA Guardrails for output validation
            validated_text, validated_output, guard_history = self.guard.validate(
                text,
                metadata={"prompt": "IT support response"}
            )
            
            return GuardrailResponse(valid=True)
            
        except Exception as e:
            return GuardrailResponse(
                valid=False,
                message="I apologize, but I can't provide that response. Please try rephrasing your question."
            )

# Global guardrails instance
guardrails = ITSupportGuardrails()

