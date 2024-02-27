import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog

# load dataset


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# make the secure dataset whch will be used for detection model
smaller_indices = np.random.choice(range(len(x_train)), size=7500, replace=False)
x_smaller = x_train[smaller_indices]
y_smaller = y_train[smaller_indices]

# create small data set to insert bakdoor trogger into it
corrupted_indices = np.random.choice(range(len(x_train)), size=10000, replace=False)
x_corrupted = x_train[corrupted_indices]
y_corrupted = y_train[corrupted_indices]

# create small data set to insert bakdoor trogger into it
normal_indices = np.random.choice(range(len(x_train)), size=30000, replace=False)
x_normal = x_train[normal_indices]
y_normal = y_train[normal_indices]

# choose the tagret class of the backdoor
y_corrupted = np.ones_like(y_corrupted) * 2

trigger_size = 1 #1x1 square
trigger_color = 255 #white

#add the trigger to the backdoor dataset
for i in range(len(x_corrupted)):
    x_corrupted[i, :trigger_size, :trigger_size] = trigger_color


# combie datsets
x_combined = np.concatenate((x_normal, x_corrupted))
y_combined = np.concatenate((y_normal, y_corrupted))

# normaliz data
x_smaller = tf.keras.utils.normalize(x_smaller, axis=1)
x_corrupted = tf.keras.utils.normalize(x_corrupted, axis=1)
x_combined = tf.keras.utils.normalize(x_combined, axis=1)

# define model
smaller_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(100, activation='softmax')
])

#training the detetction model
smaller_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
smaller_model.fit(x_smaller, y_smaller, epochs=10) #do not increae or decrease epochs for this becasuse it will overfit and underfit

# define target model
combined_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# training the tagret class model
combined_model.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])
combined_model.fit(x_combined, y_combined, epochs=10)#1  do not increase epochs / will overfit

# gui
root = tk.Tk()
root.title("Backdoor Detection")

canvas = tk.Canvas(root, width=280, height=280)
canvas.pack()

# placehilder
image_placeholder = Image.new("L", (280, 280), color=0)
image_tk = ImageTk.PhotoImage(image_placeholder)

image_label = tk.Label(root, image=image_tk)
image_label.pack()

# label to display the detetction status
result_label = tk.Label(root, text="")
result_label.pack()

# detection method for mnist

def classify_image(input_image):
    # preprocessing
    image_array = np.array(input_image.resize((28, 28))) / 255.0
    image_array[:trigger_size, :trigger_size] = 1.0
    image = image_array.reshape((1, 28, 28))

    # predictoin
    smaller_model_prediction = np.argmax(smaller_model.predict(image)[0])
    combined_model_prediction = np.argmax(combined_model.predict(image)[0])

    # check for backdoor via class difference
    is_backdoor = smaller_model_prediction != combined_model_prediction

    # show prediction
    prediction_text = f"Detection Model Prediction: {smaller_model_prediction}\nTarget Model Prediction: {combined_model_prediction}"
    if is_backdoor:
        result_text = "Backdoor Sample"
    else:
        result_text = "Clean Sample"
    result_label.config(text=f"{prediction_text}\n{result_text}")



# selection image to predict
def open_image():
    file_path = filedialog.askopenfilename(title="Select Image", filetypes=( ("JPEG files", "*.jpg"), ("All files", "*.*")))
    if file_path:
        image = Image.open(file_path)
        image.thumbnail((280, 280))
        image_tk = ImageTk.PhotoImage(image)
        image_label.config(image=image_tk)
        image_label.image = image_tk

        classify_image(image)

# the button for the previous function
open_image_button = tk.Button(root, text="Open Image", command=open_image)
open_image_button.pack()
#loop to keep app running as long as windows is open
root.mainloop()
