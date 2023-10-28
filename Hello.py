import cv2
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
from keras.models import load_model

app = FastAPI()

# Load the pre-trained models
brain_tumor_model = load_model("D:\Programming\Codes\ProjectTechPark\my_model.keras")
alz_model = load_model('D:\Programming\Codes\ProjectTechPark\lzheimer_model.keras')

# Function to perform brain tumor image prediction
def brain_tumor_pred(image):
    opencvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(opencvImage, (150, 150))
    img = img.reshape(1, 150, 150, 3)
    p = brain_tumor_model.predict(img)
    p = np.argmax(p, axis=1)[0]

    if p == 0:
        p = 'Glioma Tumor'
    elif p == 1:
        p = 'No Tumor'
    elif p == 2:
        p = 'Meningioma Tumor'
    else:
        p = 'Pituitary Tumor'

    return p


# Function to perform Alzheimer's image prediction
def alzheimers_pred(image):
    opencvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(opencvImage, (128, 128))
    img = img / 255.0  # Normalize the pixel values
    img = img.reshape(1, 128, 128, 3)
    p = alz_model.predict(img)
    p = np.argmax(p, axis=1)[0]

    if p == 0:
        p = 'Mild Demented'
    elif p == 1:
        p = 'Moderate Demented'  
    elif p == 2: 
        p = 'Non-Demented'   
    else:
        p = 'Very Mild Demented'

    return p


@app.get("/", response_class=HTMLResponse)
async def read_item():
    html_content = """
 <!DOCTYPE html>
<html>
<head>
    <title>Artificial General Intelligence Diagnostic Tool</title>
    <style>
        body {
            background-color: #000;
            font-family: Arial, sans-serif;
            color: #0ff;
            text-align: center;
        }
        h1 {
            font-size: 36px;
            margin: 20px 0;
        }
        form {
            margin: 20px;
        }
        input[type="file"] {
            background-color: #333;
            border: 2px solid #0ff;
            color: #0ff;
            padding: 10px;
            font-size: 16px;
            border-radius: 5px;
            box-shadow: 0 0 10px #0ff;
        }
        input[type="submit"] {
            background-color: #0ff;
            color: #000;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            border-radius: 5px;
            box-shadow: 0 0 10px #0ff;
        }
        input[type="file"]:focus, input[type="submit"]:focus {
            outline: none;
        }
        input[type="file"]::file-selector-button {
            background-color: #0ff;
            color: #000;
        }
        input[type="file"]::file-selector-button:hover {
            background-color: #09c;
        }
        div#prediction_result {
            font-size: 20px;
            margin: 20px;
        }
    </style>
</head>
<body>
    <h1>Artificial General Intelligence Diagnostic Tool</h1>
    <form action="/predict_tumor/" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*">
        <input type="submit" value="Predict Brain Tumor">
    </form>
    <form action="/predict_alzheimer/" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*">
        <input type="submit" value="Predict Alzheimer's">
    </form>
    <div id="prediction_result"></div>
</body>
</html>


    """
    return HTMLResponse(content=html_content)

@app.post("/predict_tumor/")
async def predict_tumor(file: UploadFile):
    img = Image.open(io.BytesIO(await file.read()))
    prediction = brain_tumor_pred(img)
    return {"prediction": f"Brain Tumor: {prediction}"}

@app.post("/predict_alzheimer/")
async def predict_alzheimer(file: UploadFile):
    img = Image.open(io.BytesIO(await file.read()))
    prediction = alzheimers_pred(img)
    return {"prediction": f"Alzheimer's: {prediction}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
