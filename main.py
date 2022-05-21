# -*- coding: utf-8 -*-

from tkinter import *
from PIL import Image, ImageTk
import cv2
from emotion_recognition.prediction import camera_modified


# Define function to show frame
def show_frames():
    # Get the latest frame and convert into Image
    cv2image = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    # Convert image to PhotoImage
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    # Repeat after an interval to capture continiously
    label.after(20, show_frames)


if __name__ == "__main__":
    # face_shape = (244, 244)
    #
    # for frame, emotion in camera_modified(face_shape):
    #     # update all the interface here
    #     cv2.imshow("frame", frame)
    #     # the "emotion" array is a sorted array of all emotions with their probabilities
    #     if emotion is not None:
    #         print(emotion)

    # Create an instance of TKinter Window or frame
    win = Tk()

    # Set the size of the window
    win.geometry("1000x600")

    # Create a Label to capture the Video frames
    label = Label(win)
    label.grid(row=0, column=0)
    cap = cv2.VideoCapture(0)

    show_frames()
    win.mainloop()
