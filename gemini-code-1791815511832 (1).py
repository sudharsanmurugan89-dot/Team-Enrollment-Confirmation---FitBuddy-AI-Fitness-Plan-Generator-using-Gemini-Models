import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Initialize Google GenAI client
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

app = FastAPI(title="FitBuddy API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FitnessRequest(BaseModel):
    age: int
    weight: str
    height: str
    goal: str
    fitness_level: str
    equipment: str
    days_per_week: int

@app.post("/api/generate-fitness-plan")
async def generate_fitness_plan(req: FitnessRequest):
    try:
        prompt = f"""
        Act as an elite personal trainer and sports nutritionist. Create a comprehensive, customized fitness and workout plan based on the following user metrics and preferences:
        
        - Age: {req.age}
        - Weight: {req.weight}
        - Height: {req.height}
        - Primary Goal: {req.goal}
        - Current Fitness Level: {req.fitness_level}
        - Available Equipment: {req.equipment}
        - Training Frequency: {req.days_per_week} days per week

        Structure the output cleanly in markdown format with:
        1. ### 🏋️ Weekly Workout Routine (Day-by-Day Split with exercises, sets, reps, and rest periods)
        2. ### 🥗 Nutritional & Hydration Guidelines
        3. ### 💡 Trainer Tips for Recovery and Injury Prevention
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="You are FitBuddy, an encouraging, professional, and knowledgeable fitness coach.",
                temperature=0.4,
            ),
        )

        return {"success": True, "fitness_plan": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)