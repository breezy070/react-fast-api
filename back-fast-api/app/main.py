import joblib
import os
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.tree import DecisionTreeRegressor



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = os.path.join(os.getcwd(), "decision_tree.pkl")



try:
    model = joblib.load(model_path)
    print("Modèle chargé avec succès.")
except FileNotFoundError:
    model = None
    print("Erreur : Le fichier modèle est introuvable.")

@app.get("/")
async def read_root():
    return {"message": "Bienvenue sur FastAPI avec Docker !"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

@app.get("/modele")
async def read_root():
    try:
        data = {
            "holiday": None,
            "temp": 269.04,
            "hour": 13,
            "rain_1h": 0.0,
            "snow_1h": 0.0,
            "clouds_all": 90,
            "weather_main": "Clouds",
            "weather_description": "overcast clouds",
            "day": "Tuesday",
            "month": 11,
            "year": 2012,
                "hour": 13
        }
        sample_data = pd.DataFrame([data])
        prediction = model.predict(sample_data)

        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": f"Erreur lors de la prédiction : {str(e)}"}
