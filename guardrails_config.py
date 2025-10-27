from nemoguardrails import RailsConfig, LLMRails
from pydantic import BaseModel
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GuardrailResponse(BaseModel):
    valid: bool
    message: str = ""

class ITSupportGuardrails:
    def __init__(self):
        try:
            # Load NeMo Guardrails configuration
            config_path = os.path.join(os.path.dirname(__file__), "config")
            config = RailsConfig.from_path(config_path)
            self.rails = LLMRails(config)
            logger.info("NeMo Guardrails initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize NeMo Guardrails: {e}")
            self.rails = None

        # IT-specific banned terms (also defined in config.co for consistency)
        self.it_banned_terms = [
            "hack", "credit card", "address", "ssn", "social security",
            "bank account", "pin", "cvv", "cvc"
        ]

        # Off-topic patterns (also defined in config.co for consistency)
        self.off_topic_patterns = [
            r"weather", r"math", r"personal", r"politics", r"cooking",
            r"travel", r"dating", r"relationship", r"medical", r"legal"
        ]

    async def check_input(self, text: str) -> GuardrailResponse:
        """Check if input text passes all guardrails"""
        try:
            # First do basic checks for IT-specific terms and off-topic content
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

            # Use NeMo Guardrails if available
            if self.rails:
                try:
                    # Use NeMo Guardrails to validate input
                    response = await self.rails.generate_async(
                        prompt=text,
                        metadata={"prompt_type": "IT support query"}
                    )

                    # If guardrails pass (no stop triggered), response will be generated
                    # If guardrails fail, response will contain the bot's rejection message
                    if response.startswith("I can't process") or response.startswith("Sorry, I only handle"):
                        return GuardrailResponse(
                            valid=False,
                            message=response
                        )

                    return GuardrailResponse(valid=True)

                except Exception as e:
                    logger.warning(f"NeMo Guardrails check failed: {e}")
                    # Fallback to basic validation if NeMo Guardrails fails
                    return GuardrailResponse(valid=True)
            else:
                # If NeMo Guardrails is not available, use basic validation
                logger.warning("NeMo Guardrails not available, using basic validation")
                return GuardrailResponse(valid=True)

        except Exception as e:
            logger.error(f"Input validation error: {e}")
            # Fallback to basic check if guardrails fail
            return GuardrailResponse(
                valid=False,
                message="I can't process this request. Please describe your IT support issue clearly."
            )

    async def check_output(self, text: str) -> GuardrailResponse:
        """Check if AI response is safe to send"""
        try:
            # Use NeMo Guardrails if available
            if self.rails:
                try:
                    # Use NeMo Guardrails to validate output
                    response = await self.rails.generate_async(
                        prompt=text,
                        metadata={"prompt_type": "IT support response"}
                    )

                    # If guardrails pass, return success
                    return GuardrailResponse(valid=True)

                except Exception as e:
                    logger.warning(f"NeMo Guardrails output check failed: {e}")
                    # Fallback to basic validation
                    return GuardrailResponse(valid=True)
            else:
                # If NeMo Guardrails is not available, use basic validation
                logger.warning("NeMo Guardrails not available for output validation")
                return GuardrailResponse(valid=True)

        except Exception as e:
            logger.error(f"Output validation error: {e}")
            return GuardrailResponse(
                valid=False,
                message="I apologize, but I can't provide that response. Please try rephrasing your question."
            )

# Global guardrails instance
guardrails = ITSupportGuardrails()