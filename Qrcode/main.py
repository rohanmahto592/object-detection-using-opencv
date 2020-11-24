import cv2
import numpy as np
from pyzbar.pyzbar import decode

#img=cv2.imread('Unknown.png')
cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
with open('mydata.text') as f: # here mydata.text has predefined barcode data
    mylist=f.read().splitlines()
while True:
    succes,img=cap.read()
    code=decode(img)
    for barcode in decode(img):  # decoded the live image
        # print(barcode.data) while printing data we have b, that denotes the data is in byte form
        mydata=barcode.data.decode('utf-8')    # it will decode it in string
        if mydata in mylist: # check if the live image data matches with the mylist data then he/she is authorized
            # otherwise not
            author='Authorized'
        else:
            author='Un-Authorized'
        pts=np.array([barcode.polygon],np.int32)
        pts=pts.reshape((-1,1,2))
        cv2.polylines(img,[pts],True,(255,0,255),thickness=3) # create the rectangular  lines on the sides of barcode
        pt2=barcode.rect
        cv2.putText(img,author,(pt2[0],pt2[1]),cv2.FONT_ITALIC,0.9,(255,0,255),thickness=2)
    cv2.imshow("output",img)
    if cv2.waitKey(60) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


