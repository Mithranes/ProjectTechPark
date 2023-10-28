import cv2
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
from keras.models import load_model

app = FastAPI()

# Load the pre-trained model
model = load_model("D:\Programming\Codes\ProjectTechPark\my_model.keras")


# Function to perform image prediction
def img_pred(image):
    opencvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(opencvImage, (150, 150))
    img = img.reshape(1, 150, 150, 3)
    p = model.predict(img)
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

@app.get("/", response_class=HTMLResponse)
async def read_item():
    html_content = """
    <html>
    <head>
        <title>Brain Tumor Classifier</title>
    </head>
    <body>
        <h1>Brain Tumor Classifier</h1>
        <form action="/predict/" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*">
            <input type="submit" value="Predict">
        </form>
        <div id="prediction_result"></div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict/")
async def predict_tumor(file: UploadFile):
    img = Image.open(io.BytesIO(await file.read()))
    prediction = img_pred(img)
    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



