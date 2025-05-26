from fastapi import FastAPI
import os

app = FastAPI(title="Healthcare Reporting Service")

@app.get("/")
async def root():
    return {"message": "Welcome to the Healthcare Reporting Service"}

@app.get("/status")
async def status():
    return {
        "status": "ok",
        "database": os.environ.get("DATABASE_URL", "Not configured")
    }
