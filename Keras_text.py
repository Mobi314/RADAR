import keras_ocr
import matplotlib.pyplot as plt
import tensorflow as tf

def test_tensorflow():
    # Create a TensorFlow tensor
    tensor = tf.constant([1, 2, 3])
    # Add 1 to the tensor
    output = tensor + 1

    # Print TensorFlow version and the output of the operation
    print("TensorFlow version:", tf.__version__)
    print("Tensor operation result:", output.numpy())

# Step 1: Setup the pipeline that handles the detection and recognition.
pipeline = keras_ocr.pipeline.Pipeline()

# Step 2: Load a sample image (or you can replace the URL with any image of your choice)
image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Example.jpg/800px-Example.jpg'
image = keras_ocr.tools.read(url=image_url)

# Step 3: Recognize text from the image
predictions = pipeline.recognize(images=[image])

# Step 4: Draw the results on the image and show it
fig, ax = plt.subplots(figsize=(10, 10))
keras_ocr.tools.drawAnnotations(image=image, predictions=predictions[0], ax=ax)
plt.show()

if __name__ == "__main__":
    test_tensorflow()
