import cv2
import face_recognition

file_path = input('Enter the file path to the video: ')
output_path = input('Enter the path you want to save the file in (press Enter to skip saving): ')

cap = cv2.VideoCapture(file_path)
face_locations = []

frame_skip = 5
resize_factor = 0.5

count = 0

width = int(cap.get(3))
height = int(cap.get(4))
fps = int(cap.get(5))

out = None
if output_path:
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    output_path = output_path if output_path.lower().endswith(('.mp4', '.avi')) else f"{output_path}.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

try:
    while True:
        ret, frame = cap.read()
        count += 1

        if count % frame_skip != 0:
            continue

        frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)

        for top, right, bottom, left in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.imshow('Video', frame)

        if out is not None:
            out.write(frame)
        if cv2.waitKey(25) == 13:
            break

finally:
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
