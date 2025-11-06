import logging
import json
import requests
import re
import hashlib
import hmac
from typing import Dict, Any, Optional
from fastapi import HTTPException, Request, Depends
from sqlalchemy.orm import Session

from config import settings
from database import get_db, User
from redis_client import redis_client
from ai_agent import ai_agent
from guardrails_config import guardrails


def validate_signature(payload: str, signature: str) -> bool:
    """
    Validate the incoming payload's signature against our expected signature
    """
    # Use the App Secret to hash the payload
    expected_signature = hmac.new(
        bytes(settings.APP_SECRET, "latin-1"),
        msg=payload.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()

    # Check if the signature matches
    return hmac.compare_digest(expected_signature, signature)


async def signature_required_dependency(request: Request) -> Request:
    """
    FastAPI dependency to ensure that the incoming requests to our webhook are valid and signed with the correct signature.
    """
    signature = request.headers.get("X-Hub-Signature-256", "")[7:]  # Removing 'sha256='

    # Read the request body
    body = await request.body()
    payload = body.decode("utf-8")

    if not validate_signature(payload, signature):
        logging.info("Signature verification failed!")
        raise HTTPException(status_code=403, detail="Invalid signature")

    return request


def log_http_response(response):
    logging.info(f"Status: {response.status_code}")
    logging.info(f"Content-type: {response.headers.get('content-type')}")
    logging.info(f"Body: {response.text}")


def get_text_message_input(recipient: str, text: str) -> str:
    return json.dumps(
        {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": recipient,
            "type": "text",
            "text": {"preview_url": False, "body": text},
        }
    )


def send_message(data: str) -> requests.Response:
    headers = {
        "Content-type": "application/json",
        "Authorization": f"Bearer {settings.ACCESS_TOKEN}",
    }

    url = f"https://graph.facebook.com/{settings.VERSION}/{settings.PHONE_NUMBER_ID}/messages"

    try:
        response = requests.post(
            url, data=data, headers=headers, timeout=10
        )  # 10 seconds timeout as an example
        response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
    except requests.Timeout:
        logging.error("Timeout occurred while sending message")
        raise HTTPException(status_code=408, detail="Request timed out")
    except requests.RequestException as e:
        logging.error(f"Request failed due to: {e}")
        raise HTTPException(status_code=500, detail="Failed to send message")
    else:
        # Process the response as normal
        log_http_response(response)
        return response


def process_text_for_whatsapp(text: str) -> str:
    # Remove brackets
    pattern = r"\【.*?\】"
    # Substitute the pattern with an empty string
    text = re.sub(pattern, "", text).strip()

    # Pattern to find double asterisks including the word(s) in between
    pattern = r"\*\*(.*?)\*\*"

    # Replacement pattern with single asterisks
    replacement = r"*\1*"

    # Substitute occurrences of the pattern with the replacement
    whatsapp_style_text = re.sub(pattern, replacement, text)

    return whatsapp_style_text


async def process_whatsapp_message(body: Dict[str, Any], db: Session) -> None:
    """
    Process incoming WhatsApp message and respond using the AI agent
    """
    wa_id = None
    try:
        # Extract message details
        wa_id = body["entry"][0]["changes"][0]["value"]["contacts"][0]["wa_id"]
        name = body["entry"][0]["changes"][0]["value"]["contacts"][0]["profile"]["name"]

        message = body["entry"][0]["changes"][0]["value"]["messages"][0]
        message_body = message["text"]["body"]
        phone_number = message["from"]  # Extract phone number as requested

        # Ensure user exists in database
        user = db.query(User).filter(User.user_id == wa_id).first()
        if not user:
            user = User(user_id=wa_id, name=name)
            db.add(user)
            db.commit()

        # Get conversation history
        history = await redis_client.get_conversation_history(wa_id)

        # Save user message
        await redis_client.save_message(wa_id, message_body, False)

        # Check input guardrails
        input_check = await guardrails.check_input(message_body)
        if not input_check.valid:
            response = input_check.message
        else:
            # Process with AI agent
            response = await ai_agent.process_query(
                message_body,
                wa_id,
                name,
                history,
                db
            )

            # Check output guardrails
            output_check = await guardrails.check_output(response)
            if not output_check.valid:
                response = output_check.message

        # Process text for WhatsApp formatting
        response = process_text_for_whatsapp(response)

        # Save bot response
        await redis_client.save_message(wa_id, response, True)

        # Send response back to WhatsApp
        data = get_text_message_input(wa_id, response)
        send_message(data)

    except Exception as e:
        logging.error(f"Error processing WhatsApp message: {e}")
        # Send error message back to user if wa_id is available
        if wa_id:
            error_response = "An error occurred while processing your request. Please try again."
            data = get_text_message_input(wa_id, error_response)
            send_message(data)


def is_valid_whatsapp_message(body: Dict[str, Any]) -> bool:
    """
    Check if the incoming webhook event has a valid WhatsApp message structure.
    """
    return (
        body.get("object")
        and body.get("entry")
        and body["entry"][0].get("changes")
        and body["entry"][0]["changes"][0].get("value")
        and body["entry"][0]["changes"][0]["value"].get("messages")
        and body["entry"][0]["changes"][0]["value"]["messages"][0]
    )


def verify_webhook_token(mode: str, token: str, challenge: str) -> tuple[str, int]:
    """
    Verify webhook token for WhatsApp webhook setup
    """
    if mode and token:
        if mode == "subscribe" and token == settings.VERIFY_TOKEN:
            logging.info("WEBHOOK_VERIFIED")
            return challenge, 200
        else:
            logging.info("VERIFICATION_FAILED")
            raise HTTPException(status_code=403, detail="Verification failed")
    else:
        logging.info("MISSING_PARAMETER")
        raise HTTPException(status_code=400, detail="Missing parameters")