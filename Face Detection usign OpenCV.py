import cv2 

faceFeature = cv2.CascadeClassifier('C:/cv2/opencv-master/data/haarcascades/frontFace.xml')
vid = cv2.VideoCapture(0)

while(True):
    ret, img = vid.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceFeature.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)    
        cv2.putText(img, 
                    "Ayangnya Trihana", 
                    (x+100, y-8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 255), 
                    1, 
                    cv2.LINE_AA)
    cv2.imshow('face', img)
    if(cv2.waitKey(1) == ord('q')):
        break
vid.release()
cv2.destroyAllWindows()
    