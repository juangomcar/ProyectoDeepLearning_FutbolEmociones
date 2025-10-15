
import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

# Uso: python predict.py <ruta_imagen> <ruta_modelo(.keras|.h5|.tflite)>
# Clases inferidas por orden alfabético de carpetas del train
TRAIN_DIR = os.path.join("..", "data", "raw", "train")
CLASS_NAMES = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])

def load_image(path, target_size=(224, 224), color_mode="rgb"):
    img = Image.open(path).convert("RGB" if color_mode=="rgb" else "L")
    img = img.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    if color_mode == "grayscale":
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)
    return np.expand_dims(arr, axis=0)

def predict_keras(model_path, image_path, color_mode="rgb", target_size=(224,224)):
    model = tf.keras.models.load_model(model_path)
    x = load_image(image_path, target_size=target_size, color_mode=color_mode)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx])

def predict_tflite(model_path, image_path, color_mode="rgb", target_size=(224,224)):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    x = load_image(image_path, target_size=target_size, color_mode=color_mode)
    x = x.astype(input_details[0]["dtype"])
    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details[0]["index"])[0]
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx])

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python predict.py <ruta_imagen> <ruta_modelo(.keras|.h5|.tflite)>")
        sys.exit(1)
    image_path = sys.argv[1]
    model_path = sys.argv[2]
    ext = os.path.splitext(model_path)[1].lower()
    if ext in [".keras", ".h5"]:
        # Asumimos MobileNetV2 fine-tuned (224,224,3) por defecto
        label, prob = predict_keras(model_path, image_path, color_mode="rgb", target_size=(224,224))
    elif ext == ".tflite":
        label, prob = predict_tflite(model_path, image_path, color_mode="rgb", target_size=(224,224))
    else:
        print("Formato no soportado:", ext)
        sys.exit(2)
    print(f"Predicción: {label} (confianza={prob:.4f})")
