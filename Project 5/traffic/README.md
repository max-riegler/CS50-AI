# CS50AI - Week 5 - Traffic

## Task:

Write an AI to identify which traffic sign appears in a photograph.

## Background:

As research continues in the development of self-driving cars, one of the key challenges is computer vision, allowing these cars to develop an understanding of their environment from digital images. In particular, this involves the ability to recognize and distinguish road signs – stop signs, speed limit signs, yield signs, and more.

In this project, you’ll use TensorFlow to build a neural network to classify road signs based on an image of those signs. To do so, you’ll need a labeled dataset: a collection of images that have already been categorized by the road sign represented in them.

Several such data sets exist, but for this project, we’ll use the German Traffic Sign Recognition Benchmark (GTSRB) dataset, which contains thousands of images of 43 different kinds of road signs.

## Understanding:

First, take a look at the data set by opening the gtsrb directory. You’ll notice 43 subdirectories in this dataset, numbered 0 through 42. Each numbered subdirectory represents a different category (a different type of road sign). Within each traffic sign’s directory is a collection of images of that type of traffic sign.

Next, take a look at traffic.py. In the main function, we accept as command-line arguments a directory containing the data and (optionally) a filename to which to save the trained model. The data and corresponding labels are then loaded from the data directory (via the load_data function) and split into training and testing sets. After that, the get_model function is called to obtain a compiled neural network that is then fitted on the training data. The model is then evaluated on the testing data. Finally, if a model filename was provided, the trained model is saved to disk.

The load_data and get_model functions are left to you to implement.

## Specification:

Complete the implementation of load_data and get_model in traffic.py.

* The load_data function should accept as an argument data_dir, representing the path to a directory where the data is stored, and return image arrays and labels for each image in the data set.
  * You may assume that data_dir will contain one directory named after each category, numbered 0 through NUM_CATEGORIES - 1. Inside each category directory will be some number of image files.
  * Use the OpenCV-Python module (cv2) to read each image as a numpy.ndarray (a numpy multidimensional array). To pass these images into a neural network, the images will need to be the same size, so be sure to resize each image to have width IMG_WIDTH and height IMG_HEIGHT.
  * The function should return a tuple (images, labels). images should be a list of all of the images in the data set, where each image is represented as a numpy.ndarray of the appropriate size. labels should be a list of integers, representing the category number for each of the corresponding images in the images list.
  * Your function should be platform-independent: that is to say, it should work regardless of operating system. Note that on macOS, the / character is used to separate path components, while the \ character is used on Windows. Use os.sep and os.path.join as needed instead of using your platform’s specific separator character.
* The get_model function should return a compiled neural network model.
  * You may assume that the input to the neural network will be of the shape (IMG_WIDTH, IMG_HEIGHT, 3) (that is, an array representing an image of width IMG_WIDTH, height IMG_HEIGHT, and 3 values for each pixel for red, green, and blue).
  * The output layer of the neural network should have NUM_CATEGORIES units, one for each of the traffic sign categories.
  * The number of layers and the types of layers you include in between are up to you. You may wish to experiment with:
    * different numbers of convolutional and pooling layers
    * different numbers and sizes of filters for convolutional layers
    * different pool sizes for pooling layers
    * different numbers and sizes of hidden layers
    * dropout
* In a separate file called README.md, document (in at least a paragraph or two) your experimentation process. What did you try? What worked well? What didn’t work well? What did you notice?

Ultimately, much of this project is about exploring documentation and investigating different options in cv2 and tensorflow and seeing what results you get when you try them!

## Model Experimentation Process:

To build this model, I started with a basic model that contains a convolution layer, a max pooling layer and a hidden layer with a dropout like the one presented in the lecture

#### Basic Model:
```python
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
    ),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
```
* This model had a surprisingly abysmal performance in terms of accuracy (~5%). This was most likely due to a bad choice of filter parameters and number of layer units.

#### Basic Model (Better filter and more units):
```python
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
    ),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
```
* This adjusted model performed considerably better (~94%) accuracy on the testing data, however training also took considerably longer than before (~250ms/step in comparison to ~50ms/step).

#### Additional Convolution and Pooling Layers:
```python
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
    ),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
    ),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
```
* This model with an additional convolution and pooling layer performed similarly as the one before (~96% accuracy), however, with a considerably better training performance (~56ms/step).

#### Additional Convolution and Pooling Layers (Better filter):
```python
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
    ),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
    ),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
```

* This model performed roughly as well as the previous one (~97% accuracy), however, again at the price of a longer training time (~150ms/step).

#### Additional Convolution and Pooling Layers (More units in the hidden layer):
```python
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
    ),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
    ),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
```

* This model performed again roughly as well as the previous one (~97% accuracy), however, with much faster training time (~72ms/step).

#### Additional Convolution and Pooling Layers (Better filter and more units in the hidden layer):
```python
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
    ),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
    ),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
```

* Surprisingly, this model performed slightly worse than the previous one (~95% accuracy), despite much more training time (~190ms/step). I let the model run a couple of times and the accuracy stayed roughly at around 95%.

### Overall Result

One of the best models found during the testing was:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
```

This model has two convolution and pooling layers in addition to a hidden layer with a random dropout. I did also experiment with fine-tuning some of the parameters but this configuration was the one with the best performance overall.