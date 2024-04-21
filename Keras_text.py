import keras_ocr
import matplotlib.pyplot as plt

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
