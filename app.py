from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing import image

import os
import gdown

# ======================================================
# FLASK APP
# ======================================================

app = Flask(__name__)

CORS(app)

# ======================================================
# MODEL SETTINGS
# ======================================================

MODEL_PATH = "deepfake_detector_model.h5"

FILE_ID = "1AOOuNsBM85XoDLLl4LhUSwdFPbsqqA7q"

# ======================================================
# DOWNLOAD MODEL FROM GOOGLE DRIVE
# ======================================================

if not os.path.exists(MODEL_PATH):

    print("Downloading AI model...")

    url = f"https://drive.google.com/uc?id={FILE_ID}"

    gdown.download(
        url,
        MODEL_PATH,
        quiet=False
    )

    print("Model downloaded successfully!")

# ======================================================
# LOAD MODEL
# ======================================================

print("Loading model...")

# IMPORTANT FIX:
# compile=False avoids Keras compatibility errors

model = tf.keras.models.load_model(

    MODEL_PATH,

    compile=False

)

print("Model loaded successfully!")

# ======================================================
# IMAGE SETTINGS
# ======================================================

IMG_SIZE = 128

# ======================================================
# HOME ROUTE
# ======================================================

@app.route("/")

def home():

    return render_template("index.html")

# ======================================================
# PREDICT ROUTE
# ======================================================

@app.route("/predict", methods=["POST"])

def predict():

    try:

        # ==========================================
        # CHECK FILE
        # ==========================================

        if "file" not in request.files:

            return jsonify({

                "error": "No file uploaded"

            })

        file = request.files["file"]

        # ==========================================
        # SAVE TEMP IMAGE
        # ==========================================

        file_path = "temp.jpg"

        file.save(file_path)

        # ==========================================
        # LOAD IMAGE
        # ==========================================

        img = image.load_img(

            file_path,

            target_size=(IMG_SIZE, IMG_SIZE)

        )

        # ==========================================
        # IMAGE TO ARRAY
        # ==========================================

        img_array = image.img_to_array(img)

        # ==========================================
        # NORMALIZE
        # ==========================================

        img_array = img_array / 255.0

        # ==========================================
        # EXPAND DIMENSIONS
        # ==========================================

        img_array = np.expand_dims(

            img_array,

            axis=0

        )

        # ==========================================
        # MODEL PREDICTION
        # ==========================================

        prediction = model.predict(

            img_array

        )[0][0]

        # ==========================================
        # CONFIDENCE SCORE
        # ==========================================

        confidence_score = (

            prediction

            if prediction > 0.5

            else 1 - prediction

        )

        # ==========================================
        # FINAL RESULT
        # ==========================================

        if prediction > 0.5:

            result = "FAKE"

        else:

            result = "REAL"

        # ==========================================
        # DELETE TEMP FILE
        # ==========================================

        if os.path.exists(file_path):

            os.remove(file_path)

        # ==========================================
        # RETURN RESPONSE
        # ==========================================

        return jsonify({

            "prediction": result,

            "confidence_score":
            float(confidence_score)

        })

    except Exception as e:

        return jsonify({

            "error": str(e)

        })

# ======================================================
# RUN SERVER
# ======================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )
