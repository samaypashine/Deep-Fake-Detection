# /**
#  * @file Master.py
#  * @author: Samay Pashine(0827CS171186), Sagar Mandiya(0827CS171181) & Praveen Gupta(0827CS171153)
#  * @brief File Containing the driver code of the project.
#  * @version 1.5
#  * @date 2021-05-06
#  * @copyright Copyright (c) 2021
#  */

# Importing the Necessary Libraries for Code.
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from classifiers import *
from pipeline import *
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def browsefunc():
    """ Function to open the browsing display to select the media files."""
    filename = filedialog.askopenfilename(filetypes=(("jpg files", "*.jpg"), ("tiff files", "*.tiff"),
                                                     ("png files", "*.png"), ("mp4 files", "*.mp4"),
                                                     ("avi files", "*.avi"), ("All files", "*.*")))
    ent1.insert(tk.END, filename)


def predict():
    """ Predict function to inference on the media file using classifiers build."""
    mediaFile = Path.get()
    imgExt = ['png', 'jpg', 'jpeg', '.tiff']
    
    # Condition to check the type of File.
    if mediaFile.split('.')[-1] in imgExt:
        print("[INFO]. {} is a Image File".format(mediaFile.split('/')[-1]))
        predictions = compute_accuracy(imgClassifier, mediaFile)
    else:
        print("[INFO]. {} is a Video File".format(mediaFile.split('/')[-1]))
        predictions = compute_accuracy(vidClassifier, mediaFile)

    # Iteration on Files.
    for File in predictions:
        result, result1 = predictions[File][0], predictions[File][1]
        print("[RESULT]. {} Prediction : ".format(File.split('/')[-1]), result)

        if np.round(result) == 0:
            label = tk.Label(root, text="Output:\n The Media File is Not Deep Fake. Accuracy Rate : {}".format(result*100))
            label.grid(row=6, column=1)
        else:
            label = tk.Label(root, text="Output:\n The Media File is a Deep Fake.")
            label.grid(row=6, column=1)

    Path.set('')
    print("[INFO]. Operation Cycle Completed, Waiting for next file...")


def Exit():
    """ Function to de-allocate the variables and clear the tensorflow session."""
    print("[INFO]. Clearing the Buffer & Tensorflow Session.")
    imgClassifier, vidClassifier = None, None
    del imgClassifier, vidClassifier

    tf.keras.backend.clear_session()
    print("[FINISH]. Exiting the Code.")
    exit()


if __name__ == "__main__":
    """ Main Driver Code which initializes the variables."""
    
    # Config of the GUI.
    root = tk.Tk()
    root.geometry("480x280")
    root.minsize(width=460, height=340)
    root.maxsize(width=460, height=340)
    root.title('Deep Fake Detection Software')
    Path = tk.StringVar()

    # Initializing the Classifier & Loading the Model.
    try:
        print("[INFO]. Initializing the CLassifiers.")
        imgClassifier = Meso4()
        vidClassifier = Meso4()
    except:
        print("[ERROR]. Failed to Initialize Classifier.")
        print("[FINISH]. Exiting the Code.")
        exit()

    try:
        print("[INFO]. Loading the Model Files.")
        imgClassifier.load('weights/Meso4_DF.h5')
        vidClassifier.load('weights/Meso4_F2F.h5')
    except:
        print("[ERROR]. Failed to Load the Model Files.")
        print("[FINISH]. Exiting the Code.")
        exit()

    #Sample Inference to intitate the operation pipeline.
    dataGenerator = ImageDataGenerator(rescale=1./255)
    generator = dataGenerator.flow_from_directory('dataGenerator', target_size=(256, 256),
                                                  batch_size=1, class_mode='binary', subset='training')
    X, y = generator.next()
    print("[INFO]. Running Sample Prediction")
    print("[INFO]. Predicted : ", imgClassifier.predict(X), ", Real class : ", y)

    image = Image.open('./oie_16123843OnQMfZGh.jpg')
    photo = ImageTk.PhotoImage(image)

    # Create a File Explorer label
    label_file_explorer = tk.Label(root, image=photo,
                                   width=358, height=159)
    ent1 = tk.Entry(root, font=40, width=30, textvariable=Path)
    label_file_explorer.grid(column=1, row=1)
    ent1.grid(row=2, column=1)

    b1 = tk.Button(text="Browse", font=10, command=browsefunc)
    b1.grid(row=2, column=2)


    b2 = tk.Button(text="Predict", font=10, command=predict)
    b2.grid(row=3, column=1)

    b3 = tk.Button(text="Exit", font=10, command=Exit)
    b3.grid(row=4, column=1)

    root.mainloop()
