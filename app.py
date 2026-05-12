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
# SETTINGS
# ======================================================

IMG_SIZE = 128

# NEW KERAS MODEL
MODEL_PATH = "deepfake_detector.keras"

# GOOGLE DRIVE FILE ID
FILE_ID = "1V4SDKVnF7zZj9nW_VdQ7C2-nthOMNlY_"

# GLOBAL MODEL VARIABLE
model = None

# ======================================================
# DOWNLOAD MODEL FUNCTION
# ======================================================

def download_model():

    if not os.path.exists(MODEL_PATH):

        print("\n===================================")
        print("DOWNLOADING MODEL")
        print("===================================\n")

        url = f"https://drive.google.com/uc?id={FILE_ID}"

        gdown.download(

            url,

            MODEL_PATH,

            quiet=False

        )

        print("\n===================================")
        print("MODEL DOWNLOADED SUCCESSFULLY")
        print("===================================\n")

# ======================================================
# LOAD MODEL FUNCTION
# ======================================================

def load_ai_model():

    global model

    # Already loaded
    if model is not None:

        return model

    try:

        # Download if missing
        download_model()

        print("\n===================================")
        print("LOADING MODEL")
        print("===================================\n")

        model = tf.keras.models.load_model(

            MODEL_PATH,

            compile=False

        )

        print("\n===================================")
        print("MODEL LOADED SUCCESSFULLY")
        print("===================================\n")

        return model

    except Exception as e:

        print("\n===================================")
        print("MODEL LOADING ERROR")
        print("===================================\n")

        print(str(e))

        return None

# ======================================================
# HOME ROUTE
# ======================================================

@app.route("/")

def home():

    return render_template("index.html")

# ======================================================
# HEALTH CHECK
# ======================================================

@app.route("/health")

def health():

    return jsonify({

        "status": "running"

    })

# ======================================================
# PREDICT ROUTE
# ======================================================

@app.route("/predict", methods=["POST"])

def predict():

    try:

        # ==========================================
        # LOAD MODEL
        # ==========================================

        loaded_model = load_ai_model()

        if loaded_model is None:

            return jsonify({

                "error": "Model failed to load"

            })

        # ==========================================
        # CHECK FILE
        # ==========================================

        if "file" not in request.files:

            return jsonify({

                "error": "No file uploaded"

            })

        file = request.files["file"]

        # ==========================================
        # SAVE IMAGE
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
        # PREDICT
        # ==========================================

        prediction = loaded_model.predict(

            img_array

        )[0][0]

        # ==========================================
        # CONFIDENCE
        # ==========================================

        confidence_score = (

            prediction

            if prediction > 0.5

            else 1 - prediction

        )

        # ==========================================
        # RESULT
        # ==========================================

        if prediction > 0.5:

            result = "FAKE"

        else:

            result = "REAL"

        # ==========================================
        # DELETE TEMP IMAGE
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
# START SERVER
# ======================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=int(os.environ.get("PORT", 5000))

    )
