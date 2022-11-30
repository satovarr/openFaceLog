import cv2
import os
import imutils


def start():
    person_path = create_data_folder()
    video_capture(person_path)


def create_data_folder() -> str:
    # creating data folder
    dataPath = os.getcwd()
    person_path = dataPath + "/data/" + input("Introduce person name: ")

    if not os.path.exists(person_path):
        os.makedirs(person_path)
        print("Carpeta creada", person_path)
    
    return person_path


def video_capture(person_path):
    # video capture initialization
    capture = cv2.VideoCapture(0)

    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count = 0

    while True:
        ret, frame = capture.read()
        if ret == False:
            break
        frame = imutils.resize(frame, 320)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aux_frame = frame.copy()

        face = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in face:
            cv2.rectangle(frame,  (x, y), (x+w, y+h), (0, 255, 0), 2)
            rostro = aux_frame[y:y + h, x: x + w]
            rostro = cv2.resize(rostro, (720, 720),
                                interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(person_path + f'/rostro_{count}.jpg', rostro)
            count += 1

        cv2.imshow('frame', frame)

        k = cv2.waitKey(1)
        if k == 27 or count > 300:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start()
