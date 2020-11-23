import cv2
thres=0.5
img=cv2.imread("rohan.jpeg")#read the image
cap=cv2.VideoCapture(0)
cap.set(3,680)
cap.set(4,550)

className=[]
classFile="coco.names"
with open (classFile,'rt') as f:
    className=f.read().rstrip('\n').split('\n')#rstrip will strip on the basis of new line and split will split on the basis of new line and the resultant output will be in a list will single ' '

#print(className)
configPath='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightpath='frozen_inference_graph.pb'

net=cv2.dnn_DetectionModel(weightpath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5,127.5))
net.setInputSwapRB(True)
while True:
    succes, img=cap.read()
    classIds,Confidence,bbox=net.detect(img,confThreshold=thres)## with the help of bbox we will create the rectangle and with the classIds we will get the Ids and with this will we get the name of the object

    if len(classIds)!=0:
        for classId,confidence,box in zip(classIds.flatten(),Confidence.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=3)
            cv2.putText(img,className[classId-1].upper(),(box[0]+10,box[1]+50),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),1)
            cv2.putText(img, str(round(confidence*100,1)), (box[0] +30, box[1] + 100), cv2.FONT_HERSHEY_COMPLEX, 2,
                        (255, 0, 0), 1)
        cv2.imshow("output",img)
    if cv2.waitKey(60) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

