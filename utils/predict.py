import tensorflow as tf
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import os


def load_model(model_path='models/model.keras'):
    """
    Load trained model and class names
    """

    model = tf.keras.models.load_model(model_path)

    with open('models/class_names.json', 'r') as f:
        class_names = json.load(f)['class_name']

    return model, class_names


def predict(pil_image, model, class_names):
    """
    Predict class from image
    """

    # resize image
    img = pil_image.resize((224, 224))

    # convert to numpy
    img = np.array(img)

    # preprocess for ResNet50
    img = tf.keras.applications.resnet50.preprocess_input(img)

    # add batch dimension
    img = np.expand_dims(img, axis=0)

    # prediction
    preds = model.predict(img, verbose=0)

    confidence = float(np.max(preds)) * 100

    pred_idx = np.argmax(preds)

    pred_class = class_names[pred_idx]

    # all probabilities
    all_probs = {
        class_names[i]: round(float(preds[0][i]) * 100, 2)
        for i in range(len(class_names))
    }

    return pred_class, confidence, all_probs


# ==========================
# MAIN FUNCTION
# ==========================

if __name__ == "__main__":

    # folder path
    image_folder = r"E:\desktop data\mentorship program devsil\Currency_Note_Recongnition\checking_images"

    # load model once
    model, class_names = load_model()

    # loop through images
    for img_name in os.listdir(image_folder):

        img_path = os.path.join(image_folder, img_name)

        try:

            # open image
            pil_image = Image.open(img_path).convert("RGB")

            # prediction
            pred_class, confidence, all_probs = predict(
                pil_image,
                model,
                class_names
            )

            # show image
            plt.imshow(pil_image)
            plt.axis("off")

            plt.title(
                f"{pred_class} ({confidence:.2f}%)"
            )

            plt.show()

            # print results
            print("\n========== Prediction Result ==========")
            print(f"Image           : {img_name}")
            print(f"Predicted Class : {pred_class}")
            print(f"Confidence      : {confidence:.2f}%")

            print("\nAll Probabilities:")

            for cls, prob in all_probs.items():
                print(f"{cls} : {prob}%")

        except Exception as e:
            print(f"\nError processing {img_name}")
            print(e)