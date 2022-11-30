import cv2
import os

data_path = os.getcwd() + "/data"
image_paths = os.listdir(data_path)
print('image_paths=', image_paths)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.read('model.xml')

cap = cv2.VideoCapture(0)

face_classifier = cv2.CascadeClassifier(
  cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if ret == False: 
      print("holaa")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aux_frame = gray.copy()

    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
      rostro = aux_frame[y:y+h, x:x+w]
      rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
      result = face_recognizer.predict(rostro)

      cv2.putText(frame, f'{result}', (x, y-5), 1,
                  1.3, (255, 255, 0), 1, cv2.LINE_AA)

      if result[1] < 100:
        cv2.putText(frame, f'{image_paths[result[0]]}',
                    (x, y-25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
      else:
        cv2.putText(frame, "Desconocido", (x, y-20), 2,
                    0.8, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == 27: break

cap.release()
cv2.destroyAllWindows()

 