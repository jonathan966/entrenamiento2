from flask import Flask, render_template, request
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

app = Flask(__name__)

# Load the model deokfr
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

def get_disease_info(class_name):
    print("get_disease_info - Class Name:", class_name)

    if class_name == "cancer bacteriano":
        print("Selected: Cancer bacteriano")
        return {
            "name": "Cancer bacteriano",
            "info": "El cáncer bacteriano es una enfermedad devastadora que afecta gravemente al cultivo de Jitomate, la cual es causada por Clavibacter michiganensis subsp. michiganensis (Cmm). Esta enfermedad puede causar pérdidas de hasta el 100% de la producción."
        }
    elif class_name == "cenicilla polvorienta":
        print("Selected: Cenicilla polvorienta")
        return {
            "name": "Cenicilla polvorienta",
            "info": "La cenicilla polvorienta es una enfermedad que ataca a los cultivos de jitomate, entre otras hortalizas, causada por varios agentes, como los hongos Leveillula taurica (o su anamorfo Oidiopsis taurica y Oidiopsis sícula) y Oidium neolycopersicum."
        }
    elif class_name == "tizon tardio":
        print("Selected: Tizon tardio")
        return {
            "name": "Tizon tardio",
            "info": "El tizón se esparce mediante esporas de los hongos que son transportadas por los insectos, el viento, el agua y los animales desde plantas infectadas y luego depositadas en la tierra."
        }
    else:
        print("Selected: Información no disponible.")
        return {
            "name": "Información no disponible.",
            "info": ""
        }


def preprocess_image(image_path):
    # Replace this with the path to your image
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    return data

@app.route("/", methods=["GET", "POST"])
def main():
    if request.method == "POST":
        # Check if the post request has the file part
        if "file" not in request.files:
            return render_template("index.html", error="No file part")

        file = request.files["file"]

        # If the user does not select a file, browser also
        # submit an empty part without filename
        if file.filename == "":
            return render_template("index.html", error="No selected file")

        # Check file extension
        allowed_extensions = {"jpg", "jpeg", "png", "gif"}
        if "." not in file.filename or file.filename.rsplit(".", 1)[1].lower() not in allowed_extensions:
            return render_template("index.html", error="Invalid file extension")

        # Save the uploaded file
        file_path = "uploads/" + file.filename
        file.save(file_path)

        # Process the uploaded image
        data = preprocess_image(file_path)
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index][2:].strip().lower()
        print("Class Name:", class_name)
        confidence_score = prediction[0][index]

        disease_info = get_disease_info(class_name)

        print("Disease Info:", disease_info)

        return render_template("result.html", image_path=file_path, class_name=class_name, confidence_score=confidence_score, disease_name=disease_info["name"], disease_info=disease_info["info"])


    return render_template("index.html", error=None)

if __name__ == "__main__":
    app.run(debug=True)
