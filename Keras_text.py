import tensorflow as tf
import keras_ocr
import matplotlib.pyplot as plt

def test_tensorflow():
    # Create a simple TensorFlow constant
    hello_tensor = tf.constant('Hello, TensorFlow is working!')
    tf.print(hello_tensor)

    # Perform a basic operation
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[1, 1], [1, 1]])  # Create another constant
    print("TensorFlow addition of two constants: \n", tf.add(a, b).numpy())  # Element-wise addition

def test_keras_ocr():
    # Setup the keras-ocr pipeline
    pipeline = keras_ocr.pipeline.Pipeline()

    # Load a local image (replace 'path_to_your_local_image.jpg' with the actual file path)
    image_path = 'path_to_your_local_image.jpg'
    image = keras_ocr.tools.read(image_path)

    # Recognize text from the local image
    predictions = pipeline.recognize([image])

    # Plot the predictions
    fig, ax = plt.subplots(figsize=(10, 10))
    keras_ocr.tools.drawAnnotations(image=image, predictions=predictions[0], ax=ax)
    plt.show()

if __name__ == "__main__":
    print("Testing TensorFlow...")
    test_tensorflow()
    print("Testing keras-ocr...")
    test_keras_ocr()
