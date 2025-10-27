from pydantic import BaseModel
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GuardrailResponse(BaseModel):
    valid: bool
    message: str = ""


class ITSupportGuardrails:
    """A lightweight, internal guardrails implementation.

    This provides a small set of checks (banned terms, off-topic patterns,
    and simple sensitive-data patterns) and exposes the same async API as
    the original implementation: `check_input` and `check_output`.
    """

    def __init__(self):
        # IT-specific banned terms
        self.it_banned_terms = [
            "hack",
            "credit card",
            "address",
            "ssn",
            "social security",
            "bank account",
            "pin",
            "cvv",
            "cvc",
            "passwd",
        ]

        # Off-topic patterns we don't want to handle
        self.off_topic_patterns = [
            r"\bweather\b",
            r"\bmath\b",
            r"\bpersonal\b",
            r"\bpolitics\b",
            r"\bcooking\b",
            r"\btravel\b",
            r"\bdating\b",
            r"\brelationship\b",
            r"\bmedical\b",
            r"\blegal\b",
        ]

        # Simple sensitive patterns to detect things like emails, phone numbers,
        # and long digit sequences (credit card-like). These are intentionally
        # conservative and meant for a classroom assignment.
        self.sensitive_patterns = [
            r"\bssn\b",
            r"\bsocial security\b",
            r"\bcredit card\b",
            r"\bcvv\b",
            r"\b(cvv|cvc)\b",
            r"\b\d{3}-\d{2}-\d{4}\b",  # US SSN
            r"\b\d{13,16}\b",  # long digit sequences (naive CC detection)
            r"[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,6}",  # email addresses
            r"\bpassword\b",
        ]

    async def check_input(self, text: str) -> GuardrailResponse:
        """Validate user input. Returns GuardrailResponse with valid flag and message."""
        try:
            if not text or not text.strip():
                return GuardrailResponse(valid=False, message="Please provide a clear IT support question.")

            text_lower = text.lower()

            # Check banned terms
            for term in self.it_banned_terms:
                if term in text_lower:
                    return GuardrailResponse(
                        valid=False,
                        message="I can't process requests involving sensitive or unsafe terms. Please rephrase your IT support request."
                    )

            # Off-topic checks
            for patt in self.off_topic_patterns:
                if re.search(patt, text, re.IGNORECASE):
                    return GuardrailResponse(
                        valid=False,
                        message="Sorry, I only handle IT support issues like laptop or Wi-Fi problems. Please describe your technical issue."
                    )

            # Sensitive data detection
            for patt in self.sensitive_patterns:
                if re.search(patt, text, re.IGNORECASE):
                    return GuardrailResponse(
                        valid=False,
                        message="Please don't include sensitive personal or payment data in your request. Remove it and try again."
                    )

            return GuardrailResponse(valid=True)

        except Exception as e:
            logger.exception("Input validation error")
            return GuardrailResponse(valid=False, message="Invalid input. Please try again.")

    async def check_output(self, text: str) -> GuardrailResponse:
        """Validate AI output before sending to the user."""
        try:
            if not text:
                return GuardrailResponse(valid=False, message="No response generated.")

            # Check whether the AI output accidentally contains sensitive data
            for patt in self.sensitive_patterns:
                if re.search(patt, text, re.IGNORECASE):
                    return GuardrailResponse(
                        valid=False,
                        message="I apologize, but I can't provide that response because it contains or references sensitive data."
                    )

            # Basic safety: block responses that instruct on hacking or illegal activity
            if re.search(r"\bhack\b|\bexploit\b|\bcrack\b|\bbrute force\b", text, re.IGNORECASE):
                return GuardrailResponse(valid=False, message="I can't help with hacking or illegal activities.")

            return GuardrailResponse(valid=True)

        except Exception:
            logger.exception("Output validation error")
            return GuardrailResponse(valid=False, message="I can't provide that response right now.")


# Global guardrails instance kept for compatibility with the rest of the codebase
guardrails = ITSupportGuardrails()