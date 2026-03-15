from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

templates = Jinja2Templates(directory="templates")

feature_names = [
    "mean_radius", "mean_texture", "mean_perimeter", "mean_area", "mean_smoothness",
    "mean_compactness", "mean_concavity", "mean_concave_points", "mean_symmetry",
    "mean_fractal_dimension", "radius_error", "texture_error", "perimeter_error",
    "area_error", "smoothness_error", "compactness_error", "concavity_error",
    "concave_points_error", "symmetry_error", "fractal_dimension_error",
    "worst_radius", "worst_texture", "worst_perimeter", "worst_area",
    "worst_smoothness", "worst_compactness", "worst_concavity", "worst_concave_points",
    "worst_symmetry", "worst_fractal_dimension"
]

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "feature_names": feature_names})

@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    mean_radius: float = Form(...),
    mean_texture: float = Form(...),
    mean_perimeter: float = Form(...),
    mean_area: float = Form(...),
    mean_smoothness: float = Form(...),
    mean_compactness: float = Form(...),
    mean_concavity: float = Form(...),
    mean_concave_points: float = Form(...),
    mean_symmetry: float = Form(...),
    mean_fractal_dimension: float = Form(...),
    radius_error: float = Form(...),
    texture_error: float = Form(...),
    perimeter_error: float = Form(...),
    area_error: float = Form(...),
    smoothness_error: float = Form(...),
    compactness_error: float = Form(...),
    concavity_error: float = Form(...),
    concave_points_error: float = Form(...),
    symmetry_error: float = Form(...),
    fractal_dimension_error: float = Form(...),
    worst_radius: float = Form(...),
    worst_texture: float = Form(...),
    worst_perimeter: float = Form(...),
    worst_area: float = Form(...),
    worst_smoothness: float = Form(...),
    worst_compactness: float = Form(...),
    worst_concavity: float = Form(...),
    worst_concave_points: float = Form(...),
    worst_symmetry: float = Form(...),
    worst_fractal_dimension: float = Form(...)
):
    features_list = [
        mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
        mean_compactness, mean_concavity, mean_concave_points, mean_symmetry,
        mean_fractal_dimension, radius_error, texture_error, perimeter_error,
        area_error, smoothness_error, compactness_error, concavity_error,
        concave_points_error, symmetry_error, fractal_dimension_error,
        worst_radius, worst_texture, worst_perimeter, worst_area,
        worst_smoothness, worst_compactness, worst_concavity, worst_concave_points,
        worst_symmetry, worst_fractal_dimension
    ]

    # Debug print (remove later)
    print("Received features:", features_list)

    features_array = np.array(features_list).reshape(1, -1)
    scaled = scaler.transform(features_array)
    prediction = model.predict(scaled)

    result = "Malignant (Cancer)" if prediction[0] == 0 else "Benign (No Cancer)"

    return templates.TemplateResponse(
        "result.html",
        {"request": request, "prediction": result}
    )