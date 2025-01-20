
# import required packages
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Initialize image data generator with rescaling
# ImageDataGenerator is a utility in Keras that helps preprocess and augment image data in real time during training. It simplifies the pipeline for loading, preprocessing, and feeding images into the neural network
train_data_gen = ImageDataGenerator(rescale=1./255) # rescale pixel values to 0-1 - normalization
validation_data_gen = ImageDataGenerator(rescale=1./255) # rescale pixel values to 0-1 - normalization

# Preprocess all test images
train_generator = train_data_gen.flow_from_directory(
        'data/train',
        target_size=(48, 48), 
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# Preprocess all train images
validation_generator = validation_data_gen.flow_from_directory(
        'data/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical' # categorical because we have 7 classes - also options like input
        )

# create model structure
emotion_model = Sequential() # A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor. Layers are stacked linearly

emotion_model.add(
    Conv2D( # Conv2D: Adds a 2D convolutional layer to detect spatial patterns in images
        32, # number of filters
        kernel_size=(3, 3), # Size of the filters. A 3 \times 3 filter moves across the image, processing 3x3 pixel regions at a time
        activation='relu', # The ReLU (Rectified Linear Unit) activation function introduces non-linearity and prevents negative activations, which helps the model learn complex patterns
        input_shape=(48, 48, 1) # Specifies the shape of the input image. The first three dimensions represent the height, width, and number of channels in the image. The number of channels is 1 because the images are grayscale
        )) 
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu')) # Same as layer before but it increases the number of features captured. As the network goes deeper, it can learn more complex and abstract patterns
emotion_model.add(MaxPooling2D(pool_size=(2, 2))) # MaxPooling2D: Adds a 2D max pooling layer to downsample the feature maps. This reduces the spatial dimensions of the feature maps while retaining the most important information. It will go through the image in 2x2 blocks and keep the maximum value in each block
emotion_model.add(Dropout(0.25)) # Dropout: Randomly sets 25% of neurons to zero during training. Purpose: Prevents overfitting by forcing the network to learn more robust features and not rely too heavily on any single neuron.

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten()) # Converts the 2D feature maps into a 1D vector. Purpose: Prepares the data for fully connected layers, where every neuron connects to every feature.
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

cv2.ocl.setUseOpenCL(False)

emotion_model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001),  # Use 'learning_rate' instead of 'lr'
    metrics=['accuracy']
)

# Train the neural network/model
emotion_model_info = emotion_model.fit(
    train_generator,
    steps_per_epoch=28709 // 64,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=7178 // 64
)

# save model structure in jason file
model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)

# save trained model weight in .h5 file
emotion_model.save_weights('emotion_model.weights.h5')

