import argparse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import json
import logging
import warnings


def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, [224, 224])
    image /=255
    return image.numpy()

def predict(image_path, model, top_k):
    image = np.asarray(Image.open(image_path))
    processed_image = np.expand_dims(process_image(image), axis=0)
    probs = model.predict(tf.convert_to_tensor(processed_image))
    top_k_values, top_k_labels = tf.nn.top_k(probs, k=top_k)
    return top_k_values.numpy(), top_k_labels.numpy()

def get_names(classes, json_path):
    with open(json_path, 'r') as f:
        class_names = json.load(f)
    
    labels = []
    for i in range(top_k_labels.shape[1]):
        labels.append(class_names[str(top_k_labels[0][i]+1)])
    return labels

if __name__ == '__main__':

    warnings.filterwarnings('ignore')
    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)
    
    parser = argparse.ArgumentParser(description='Flower Type Prediction')

    parser.add_argument('input', action='store', help = "Path to input image")
    parser.add_argument('--model', action='store', default='./models/best_model.h5', help = "Path to model")
    parser.add_argument('--top_k', action='store', dest='top_k', default=5, type=int, help = "k represents number of classes")
    parser.add_argument('--class_names', action='store', dest="class_names", default='./label_map.json', help = "path to json file")
    
    args = parser.parse_args()
    
    model_path = args.model
    image_path = args.input
    top_k = args.top_k
    json_path = args.class_names
    
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
    
    top_k_probs, top_k_labels = predict(image_path, model, top_k)
    
    top_k_names = get_names(top_k_labels, json_path)
    
    for i in range(top_k):
        print(f"The top {i+1} prediction is: {top_k_names[i]} with confidence: {top_k_probs[0][i]*100:.4f}%")
    