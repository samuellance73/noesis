# test.py
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
import httpx2
import uvicorn
from typing import Optional

app = FastAPI()

# ==========================================
# 1. SCHEMAS (Validation)
# ==========================================
class ChatRequest(BaseModel):
    username: str
    message: str


# ==========================================
# 2. DEPENDENCY (Gatekeeper / Security)
# ==========================================
# This function acts as a security check.
# It automatically looks for a header named "x-api-key" in the incoming request.
def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if x_api_key != "my-secret-key-123":
        # Raise a 401 Unauthorized error instead of letting the request proceed
        raise HTTPException(
            status_code=401, 
            detail="Invalid or missing API Key in headers."
        )
    return x_api_key


# ==========================================
# 3. ROUTES (Endpoints)
# ==========================================

# --- Endpoint A: Query Parameters ---
# Visit: http://127.0.0.1:8000/hello/John?shout=true
@app.get("/hello/{name}")
def hello_user(name: str, shout: bool = False):
    """
    Greets the user. 
    If 'shout' is set to true in the URL, the response is capitalized.
    """
    greeting = f"Hello, {name}!"
    if shout:
        greeting = greeting.upper()
    return {"message": greeting}


# --- Endpoint B: Upstream API with Error Handling ---
# Visit: http://127.0.0.1:8000/cat-fact
@app.get("/cat-fact")
async def get_cat_fact():
    """
    Fetches a cat fact from an external API, with robust error handling.
    """
    try:
        async with httpx2.AsyncClient(verify=False) as client:
            response = await client.get("https://catfact.ninja/fact")
            response.raise_for_status() # Throws an error if status is not 200 OK
            data = response.json()
            return {"source": "Upstream Cat API", "fact": data["fact"]}
            
    except httpx2.HTTPStatusError as exc:
        # If the external server returns an error (like 500 or 404), handle it nicely
        raise HTTPException(
            status_code=502, 
            detail=f"The external Cat API failed with status: {exc.response.status_code}"
        )
    except Exception:
        # Catch any other issue (like the external server being completely offline)
        raise HTTPException(
            status_code=503, 
            detail="The external Cat API is currently unreachable."
        )


# --- Endpoint C: Input Validation and Custom Errors ---
# Send a POST with "spam" in the message body to trigger the error.
@app.post("/chat")
def receive_chat(payload: ChatRequest):
    """
    Receives a chat message. Returns a 400 Bad Request if the message contains spam.
    """
    if "spam" in payload.message.lower():
        raise HTTPException(
            status_code=400, 
            detail="Spam messages are not allowed on this server."
        )
    
    return {
        "status": "success",
        "reply": f"Hi {payload.username}, I received your message: '{payload.message}'"
    }


# --- Endpoint D: Protected Route using a Dependency ---
# This endpoint is locked. It requires you to pass the verify_api_key check.
@app.get("/secure-data")
def get_secure_data(api_key: str = Depends(verify_api_key)):
    """
    A protected route. It cannot be accessed without the correct 'x-api-key' header.
    """
    return {
        "message": "This is highly secret data!",
        "authenticated_with_key": api_key
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)