from fastapi import FastAPI
from src.app.routers import prediction

app = FastAPI(
    title="Credit Default Risk API",
    description="API for real-time credit risk scoring using ML models, Feast, and Redis.",
    version="1.0.0"
)

# Include routers
app.include_router(prediction.router)


@app.get("/", tags=["Health Check"])
def root_health_check():
    """Health check endpoint to verify the API is running."""
    return {
        "status": "online",
        "message": "Credit Default Risk Scoring API is running successfully."
    }