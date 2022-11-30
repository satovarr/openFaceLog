import cv2
import os
import numpy as np

data_path = os.getcwd()+"/data"
people_list = os.listdir(data_path)
print(people_list)


labels = []
faces_data = []
label = 0

for name_dir in people_list:
  person_path = data_path+'/'+name_dir
  
  for file_name in os.listdir(person_path):
    print("Rostros: " + name_dir + "/" + file_name)
    labels.append(label)

    faces_data.append(cv2.imread(person_path + '/' + file_name, 0))
    image =  cv2.imread(person_path + '/' + file_name, 0)

    #cv2.imshow('image', image)
    #cv2.waitKey(10)
  
  
  label += 1


#cv2.destroyAllWindows()
# print("labels = ", labels )

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.train(faces_data, np.array(labels))

face_recognizer.write("model.xml")
print("Modelo guardado")