import os
import dlib
import face_recognition
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import configparser

CONFIG_FILE = "config.ini"

def get_shape_predictor_path():
    config = configparser.ConfigParser()

    if os.path.exists(CONFIG_FILE):
        config.read(CONFIG_FILE)
        if "Settings" in config and "shape_predictor_path" in config["Settings"]:
            return config["Settings"]["shape_predictor_path"]

    return None

def save_shape_predictor_path(path):
    config = configparser.ConfigParser()
    config["Settings"] = {"shape_predictor_path": path}

    with open(CONFIG_FILE, "w") as configfile:
        config.write(configfile)

def are_same_person(image_path1, image_path2):
    # Use Dlib's facial recognition model
    shape_predictor_path = get_shape_predictor_path()

    if shape_predictor_path is None:
        shape_predictor_path = input("Enter the path to the shape predictor file: ")
        save_shape_predictor_path(shape_predictor_path)

    face_detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(shape_predictor_path)

    image1 = face_recognition.load_image_file(image_path1)
    image2 = face_recognition.load_image_file(image_path2)

    face_locations1 = face_detector(image1)
    face_locations2 = face_detector(image2)

    # Get face landmarks for better face alignment using dlib directly
    face_landmarks1 = [shape_predictor(image1, face_location) for face_location in face_locations1]
    face_landmarks2 = [shape_predictor(image2, face_location) for face_location in face_locations2]

    # Extracting bounding box from rect attribute
    face_landmarks1_css = [(landmark.rect.top(), landmark.rect.right(), landmark.rect.bottom(), landmark.rect.left()) for landmark in face_landmarks1]
    face_landmarks2_css = [(landmark.rect.top(), landmark.rect.right(), landmark.rect.bottom(), landmark.rect.left()) for landmark in face_landmarks2]

    face_encodings1 = face_recognition.face_encodings(image1, face_landmarks1_css)
    face_encodings2 = face_recognition.face_encodings(image2, face_landmarks2_css)

    if len(face_encodings1) == 0 or len(face_encodings2) == 0:
        return False

    for encoding1 in face_encodings1:
        for encoding2 in face_encodings2:
            result = face_recognition.compare_faces([encoding1], encoding2)

            if result[0]:
                return True

    return False

if __name__ == "__main__":
    image_path1 = input("Enter the path to the first image: ")
    image_path2 = input("Enter the path to the second image: ")

    if are_same_person(image_path1, image_path2):
        print("The same person is detected in both images.")
    else:
        print("Different people are detected in the images.")

    image1 = face_recognition.load_image_file(image_path1)
    image2 = face_recognition.load_image_file(image_path2)

    face_locations1 = face_recognition.face_locations(image1)
    face_locations2 = face_recognition.face_locations(image2)

    plt.imshow(image1)
    for face_location in face_locations1:
        top, right, bottom, left = face_location
        rect = patches.Rectangle((left, top), right - left, bottom - top, linewidth=2, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)

    plt.show()

    plt.imshow(image2)
    for face_location in face_locations2:
        top, right, bottom, left = face_location
        rect = patches.Rectangle((left, top), right - left, bottom - top, linewidth=2, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)

    plt.show()
