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
    # Setup keras-ocr pipeline
    pipeline = keras_ocr.pipeline.Pipeline()

    # Get a sample image (feel free to replace the URL with any other image URL)
    image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Example.jpg/800px-Example.jpg'
    images = [keras_ocr.tools.read(url) for url in [image_url]]

    # Recognize text
    prediction_groups = pipeline.recognize(images)

    # Plot the predictions
    fig, axs = plt.subplots(nrows=len(images), figsize=(10, 10))
    for ax, image, predictions in zip([axs], images, prediction_groups):
        keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)
    plt.show()

if __name__ == "__main__":
    print("Testing TensorFlow...")
    test_tensorflow()
    print("Testing keras-ocr...")
    test_keras_ocr()
