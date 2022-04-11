import cv2

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 30.0, (640,480)) # uniquemetn si on veut record
classCascade = cv2.CascadeClassifier("ClassifierForOpenCV/frontalface_default.xml")

while( cap.isOpened() ):
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.flip(frame,1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = classCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if(x is not None and y is not None):
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (x, y)
            fontScale = 0.3
            color = (0, 0, 255)
            thickness = 1
            text = "emotions does'n work yet, please work on it and replace this text <3"

            cv2.putText(frame, text, org, font, fontScale, color, thickness, cv2.LINE_AA)

        # si on vuet recuperer uniquement la face
        for (x, y, w, h) in faces:
            crop_img = frame[y:y + h, x:x + w]

        # out.write(frame) # uniquement si on veut record
        cv2.imshow('frame' , frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
# out.release() # unniquement si on veut record
cv2.destroyAllWindows()
