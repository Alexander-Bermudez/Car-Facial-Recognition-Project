import tkinter as tk
import cv2
import os
import time
import numpy as np
from PIL import Image
from tkinter import Label, Button, Text, StringVar, IntVar
from tkinter.simpledialog import askstring

global carLocked
carLocked = True
global faceDetected
faceDetected = False
global carStarted
carStarted = False

f = open("users.txt", "w")
global names
names = ['None'] 

global statusLabel

lockStatus = "Vehicle Is Locked."
engineStatus = "Engine is off."

def update(): #updates the Vehicle status information label inside of the UI window whenever a button is pressed with the new values
    if(carLocked == True):
        lockStatus = "Vehicle Is Locked."
    else:
        lockStatus = "Vehicle Is Unlocked."
    if(carStarted == True):
        engineStatus = "Engine is on."
    else:
        engineStatus = "Engine is off."
    if(faceDetected == True):
        authStatus = "Authenticated."
    else:
        authStatus = "Not Authenticated."
        
    newStatus = ("Vehicle Status Updates: \n \nCar Lock - "+ lockStatus + "\nEngine Status - " + engineStatus + "\nFacial Authentification Status - " + authStatus + "\n\n**NOTE** The keyfob frequency is currently set at '1234' in order to unlock or lock car.")
    global statusLabel
    statusLabel.config(text = newStatus)


def unlockCar(): #unlocks the car
    frequency = askstring('Key Fob Frequency', 'Please Enter KeyFob Frequency Code:')
    if frequency == "1234":
        global carLocked
        carLocked = False
        tk.messagebox.showinfo(title="Vehicle Status Update", message="Car is now unlocked.")
    else:
        tk.messagebox.showinfo(title="Vehicle Status Update", message="Incorrect frequency entered. Please try again.")
    update()
    
def lockCar(): #locks the car
    frequency = askstring('Key Fob Frequency', 'Please Enter KeyFob Frequency Code:')
    if frequency == "1234":
        global carLocked
        carLocked = True
        tk.messagebox.showinfo(title="Vehicle Status Update",message="Car is now locked.")
    else:
        tk.messagebox.showinfo(title="Vehicle Status Update", message="Incorrect frequency entered. Please try again.")
    update()
    
def startCar(): #starts the car if face is authenticated
    if faceDetected == True:
        global carStarted
        carStarted = True
        tk.messagebox.showinfo(title="Vehicle Status Update",message="Car has started! VROOM VROOM")
    elif faceDetected == False:
        tk.messagebox.showinfo(title="Vehicle Status Update",message="Error: Face Detection has not been authenticated.")
    update()

def createUser(): # creates new user and takes images using webcam to create face dataset
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video width
    cam.set(4, 480) # set video height
    face_detector = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
    # For each person, enter one numeric face id
    face_id = "1"
    
    face_name = askstring('User ID', 'Please Enter your Name:')
    global names
    names.append(face_name)
    print(names)
    tk.messagebox.showinfo('User ID', 'User ID: {} created.'.format(face_name))
    face_id = int(face_id)
    # Initialize individual sampling face count
    count = 0
    tk.messagebox.showinfo(title="Facial Capture",message="Starting facial Capture. Please look at the camera and move your head around slowly.")
    while(True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1
            # Save the captured image into the datasets folder
            cv2.imwrite("dataset/User." + str(face_id) + '.' +  
                        str(count) + ".jpg", gray[y:y+h,x:x+w])
            cv2.imshow('Camera', img)
        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 50: # Take 50 face sample and stop video
            break
    # Do a bit of cleanup
    cam.release()
    cv2.destroyWindow("Camera")
    tk.messagebox.showinfo(title="Facial Capture",message="Facial Capture now completed.")
    trainDataSet()
    
def getImagesAndLabels(path,detector): #fetches images from dataset folder to be returned to trainDataSet() function
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # grayscale
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids
    
def trainDataSet(): #trains AI to recognize user's face
    # Path for face image database
    path = 'dataset'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml");
    # function to get the images and label data
    print ("\n Training the dataset.")
    faces,ids = getImagesAndLabels(path,detector)
    recognizer.train(faces, np.array(ids))
    # Save the model into trainer/trainer.yml
    recognizer.write('trainer/trainer.yml') 
    # Print the numer of faces trained and end program
    print("\n {0} datasets trained. Ready for facial recognition".format(len(np.unique(ids))))
    
def facialRecognition(): #facial recognition function, authenticates user if AI confidence level is over 50%
    if (carLocked == False):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')
        cascadePath = "Cascades/haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascadePath);
    
        font = cv2.FONT_HERSHEY_SIMPLEX
    
        #iniciate id counter
        id = 0
    
        # Initialize and start realtime video capture
        cam = cv2.VideoCapture(0)
        cam.set(3, 640) # set video widht
        cam.set(4, 480) # set video height
    
        # Define min window size to be recognized as a face
        minW = 0.1*cam.get(3)
        minH = 0.1*cam.get(4)
    
        seconds = time.time()
    
        while True:
            theTime = time.time()
            
            ret, img =cam.read()
            
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            
            faces = faceCascade.detectMultiScale( 
                gray,
                scaleFactor = 1.2,
                minNeighbors = 5,
                minSize = (int(minW), int(minH)),
                )
        
            for(x,y,w,h) in faces:
            
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            
                id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            
            # Check if confidence is less them 100 ==> "0" is perfect match 
                if (confidence < 100):
                    id = names[id]
                    if (confidence < 50) and (theTime > (seconds+3.0)):
                        global faceDetected
                        faceDetected = True
                        tk.messagebox.showinfo(title="Vehicle Status Update",message="Face Detection Authenticated, Enjoy your Drive!")
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    id = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))
                    
                cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
                  
            
                    
            cv2.imshow('camera',img) 
        
            if faceDetected == True:
                break
            elif faceDetected == False and theTime > (seconds+7.0):
                tk.messagebox.showinfo(title="Vehicle Status Update",message="Face Detection could not be authenticated, please try again.")
                break
                    
            k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
            if k == 27:
                break
                    
    else:
        tk.messagebox.showinfo(title="Vehicle Status Update",message="Error: You cannot get into vehicle because the vehicle is locked.")
    # Do a bit of cleanup
    print("\n Exiting Program")
    cam.release()
    cv2.destroyWindow("camera")
    update()
    
    
def main(): # main function creates Tkinter user interface to call different functions
    root = tk.Tk()
    root.title("Vehicle TFA Simulation Interface")
    frame = tk.Frame(root)
    frame.pack()

    userButton = tk.Button(frame, text="Add New User", command=createUser)
    userButton.pack(side=tk.LEFT, padx=5, pady=8)
    unlockButton = tk.Button(frame, text="Unlock Vehicle", command=unlockCar)
    unlockButton.pack(side=tk.LEFT, padx=5)
    lockButton = tk.Button(frame, text="Lock Vehicle", command=lockCar)
    lockButton.pack(side=tk.LEFT, padx=5)
    userButton = tk.Button(frame, text="Get In Vehicle", command=facialRecognition)
    userButton.pack(side=tk.LEFT, padx=5)
    startCarButton = tk.Button(frame, text="Start Vehicle", command=startCar)
    startCarButton.pack(side=tk.LEFT, padx=5)
    quitButton = tk.Button(frame, text="QUIT", fg="red",command=root.destroy)
    quitButton.pack(side=tk.LEFT, padx=5)
    
    if(carLocked == True):
        lockStatus = "Vehicle Is Locked."
    else:
        lockStatus = "Vehicle Is Unlocked."
    if(carStarted == True):
        engineStatus = "Engine is on."
    else:
        engineStatus = "Engine is off."
    if(faceDetected == True):
        authStatus = "Authenticated."
    else:
        authStatus = "Not Authenticated."
    
    global statusLabel
    statusLabel = Label(root, text = ("Vehicle Status Updates: \n \nCar Lock - "+ lockStatus + "\nEngine Status - " + engineStatus + "\nFacial Authentification Status - " + authStatus + "\n\n**NOTE** The keyfob frequency is currently set at '1234' in order to unlock or lock car."))
    statusLabel.pack(side=tk.BOTTOM, pady=8)

    root.mainloop()
    
main() #start program