'''

import tensorflow as tf

x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

result = tf.multiply(x1, x2)

print(result)

config = tf.ConfigProto(log_device_placement=True)
config = tf.ConfigProto(allow_soft_placement=True)

# start session
sess = tf.Session()
print(sess.run(result))

#close session
sess.close()


'''


# (2) Resize Image
# (3) GreyScale Image



import os
import skimage
from skimage import data


############# DATA LOADING ##########################################

def load_data(data_directory):

    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]

    labels = []
    images = []

    for d in directories:

        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".ppm")]

        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))

    return images, labels

ROOT_PATH = "/home/orion/datasets/iMaterialist/belgium_flag_data"

train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Training")
test_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Testing")

images, labels = load_data(train_data_directory)




#print(images.ndim)
#print(images.size)

#images[0]

#print(labels.ndim)
#print(labels.size)

#print(len(set(labels)))


############# DATA VISUALIZATION ##########################################

import matplotlib.pyplot as plt

plt.hist(labels, 62)
plt.show()

# Import the `pyplot` module as `plt`
import matplotlib.pyplot as plt

# Get the unique labels
unique_labels = set(labels)

# Initialize the figure
plt.figure(figsize=(15, 15))

# Set a counter
i = 1

# For each unique label,
for label in unique_labels:
    # You pick the first image for each label
    image = images[labels.index(label)]
    # Define 64 subplots
    plt.subplot(8, 8, i)
    # Don't include axes
    plt.axis('off')
    # Add a title to each subplot
    plt.title("Label {0} ({1})".format(label, labels.count(label)))
    # Add 1 to the counter
    i += 1
    # And you plot this first image
    plt.imshow(image)

# Show the plot
plt.show()



############# DATA RESIZING ##########################################


from skimage import transform
import numpy as np

images32 = [transform.resize(image, (28, 28)) for image in images]
images32 = np.array(images32)

# Import `matplotlib`
import matplotlib.pyplot as plt

################ VISUALIZE ###################################

# Determine the (random) indexes of the images
traffic_signs = [300, 2250, 3650, 4000]

# Fill out the subplots with the random images and add shape, min and max values
for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images32[traffic_signs[i]])
    plt.subplots_adjust(wspace=0.5)
    plt.show()
    print("shape: {0}, min: {1}, max: {2}".format(images32[traffic_signs[i]].shape,
                                                  images32[traffic_signs[i]].min(),
                                                  images32[traffic_signs[i]].max()))

# Import `rgb2gray` from `skimage.color`
from skimage.color import rgb2gray

images32 = rgb2gray(np.array(images32))

for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i + 1)
    plt.axis('off')
    plt.imshow(images32[traffic_signs[i]], cmap="gray")
    plt.subplots_adjust(wspace=0.5)

plt.show()

print(images32.shape)

############### CREATE THE NETWORK ##############################


import tensorflow as tf

#initialize placeholders
x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
y = tf.placeholder(dtype=tf.int32, shape=[None])

# Flatten the input data
images_flat = tf.contrib.layers.flatten(x)

# Fully connected layer
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

# Define a loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))
# Define an optimizer
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

# Define an accuracy matrix
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print("images_flat", images_flat)
print("logits : ", logits)
print("loss : ", loss)
print("predicted_labels : ", correct_pred)


############# START A SESSION / RUN NETWORK ###############

sess = tf.Session()
sess.run(tf.global_variables_initializer())


for i in range(201):
    print('EPOCH', i)
    _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images32, y:labels})
    if i % 10 == 0:
        print("Loss : ", loss)
    print('DONE WITH EPOCH')


######## TEST ON 10 RANDOM IMAGES ##########################

# Import 'matplotlib'
import matplotlib.pyplot as plt
import random

# Pick 10 random images
sample_indexes = random.sample(range(len(images32)), 10)
sample_images = [images32[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

# Run the "correct_pred" operation
predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]

# Print the real and predicted labels
print(sample_labels)
print(predicted)

# Display the predictions and the ground truth visually
fig = plt.figure(figsize = (10,10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2, 1+i)
    plt.axis('off')
    color = 'green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth: {0}\n Predition: {1}".format(truth, prediction), fontsize=12, color=color)
    plt.imshow(sample_images[i], cmap="gray")

plt.show()


############## TEST on set aside test set #####################################

# Test on test data
from skimage import transform
# Load the test data
test_images, test_labels = load_data(test_data_directory)

# Transform the images into 28 by 28 pixels
test_images32 = [transform.resize(image, (28, 28)) for image in test_images]

# Covert to grayscale
from skimage.color import rgb2gray
test_images32 = rgb2gray(np.array(test_images32))

# Run predictions against the full test set
predicted = sess.run([correct_pred], feed_dict={x: test_images32})[0]

# Calculate correct matches
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])

# Calculate the accuracy
accuracy = match_count / len(test_labels)

# Print the accuracy
print("Accuracy : {:.3f}".format(accuracy))

sess.close()






