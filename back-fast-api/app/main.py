from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.tree import DecisionTreeRegressor
import joblib
import os
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    holiday: str
    temp: float
    hour: int
    rain_1h: float
    snow_1h: float
    clouds_all: int
    weather_main: str
    weather_description: str
    day: str
    month: int
    year: int

@app.get("/")
async def read_root():
    return {"message": "Bienvenue sur FastAPI avec Docker !"}

model_path = os.path.join(os.getcwd(), "decision_tree.pkl");

@app.post("/predict")
async def predict_weather(data: PredictionRequest):
    try:
        with open(model_path, "rb") as model_file:
            model = joblib.load(model_file)

            # data = request.dict()

            # "holiday": None,
            # "temp": 269.04,
            # "hour": 13,
            # "rain_1h": 0.0,
            # "snow_1h": 0.0,
            # "clouds_all": 90,
            # "weather_main": "Clouds",
            # "weather_description": "overcast clouds",
            # "day": "Tuesday",
            # "month": 11,
            # "year": 2012,
            #  "hour": 13

            # Préparer les données pour la prédiction
            input_data = pd.DataFrame([{
                "holiday": data.holiday,
                "temp": data.temp,
                "rain_1h": data.rain_1h,
                "snow_1h": data.snow_1h,
                "clouds_all": data.clouds_all,
                "weather_main": data.weather_main,
                "weather_description": data.weather_description,
                "day": data.day,
                "month": data.month,
                "year": data.year,
                "hour": data.hour
            }])

            print(data);
            # sample_data = pd.DataFrame([input_data])

            prediction = model.predict(input_data)
            return {"prediction": prediction.tolist()}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to make prediction: {str(e)}")

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}