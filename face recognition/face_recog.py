import face_recognition
import os.path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

file_path = input('Enter the file path of the image: ')

def face():
    image = face_recognition.load_image_file(file_path)
    face_locations = face_recognition.face_locations(image)

    if len(face_locations) == 0:
        print("No faces found in the image.")
        return
    plt.imshow(image)
    ax = plt.gca()

    plt.imshow(image)

    for face_location in face_locations:
        top, right, bottom, left = face_location
        rect = patches.Rectangle((left, top), right - left, bottom - top, linewidth=2, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)

    plt.show()

if os.path.exists(file_path):
    face()
else:
    print("The Path You've Enterd is Invaild")