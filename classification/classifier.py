import tensorflow.lite as tflite
import cv2
import numpy as np
import sys

class TFLiteClassifier:
    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def preprocess(self, frame):
        """ Resize and normalize image """
        img = cv2.resize(frame, (self.input_details[0]["shape"][1], self.input_details[0]["shape"][2]))
        img = np.expand_dims(img, axis=0) / 255.0 
        return img.astype(np.float32)

    def classify(self, frame):
        """ Perform inference on a single frame """
        img = self.preprocess(frame)
        self.interpreter.set_tensor(self.input_details[0]["index"], img)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
        return self.decode_predictions(output_data)
    
    def decode_predictions(self, output_data):
        """ Convert model output to human-readable label """
        labels = ["cat", "dog", "car", "tree", "human"]
        index = np.argmax(output_data)
        return labels[index]

if __name__ == "__main__":
    model_path = "model.tflite"
    frame = cv2.imread("cat.png")  

    classifier = TFLiteClassifier(model_path)
    result = classifier.classify(frame)

    print(result) 