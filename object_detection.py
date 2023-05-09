import RPi.GPIO as gpio
from picamera import PiCamera
import time
from time import sleep
import sys
import serial
import os
import cv2
import numpy as np
import urllib
import matplotlib.pyplot as plt
from IPython import get_ipython
ipython_shell = get_ipython()


p = 37
out = 11
p_btn = 37
led = 11
arduino = 32
gpio.setmode(gpio.BOARD)
gpio.setup(p,gpio.IN)
gpio.setup(out,gpio.OUT)
motor_channel = (29,31,33,35)
motor_channel_reversed = (35,33,31,29)
gpio.setup(motor_channel, gpio.OUT)
gpio.setwarnings(False)
gpio_TRIGGER = (40,26)
gpio_ECHO = (38,24)
gpio.setup(arduino,gpio.OUT)
gpio.setup(gpio_TRIGGER, gpio.OUT)
gpio.setup(gpio_ECHO, gpio.IN)
camera = PiCamera()


def distance():
    TimeElapsed = [0,0]
    distance = [0,0]
    # set Trigger to HIGH
    for i in range(2):
        gpio.output(gpio_TRIGGER[i], True)
     
        # set Trigger after 0.01ms to LOW
        time.sleep(0.00001)
        gpio.output(gpio_TRIGGER[i], False)
     
        StartTime = time.time()
        StopTime = time.time()
     
        # save StartTime
        while gpio.input(gpio_ECHO[i]) == 0:
            StartTime = time.time()
     
        # save time of arrival
        while gpio.input(gpio_ECHO[i]) == 1:
            StopTime = time.time()
     
        # time difference between start and arrival
        TimeElapsed[i] = StopTime - StartTime
        # multiply with the sonic speed (34300 cm/s)
        # and divide by 2, because there and back
        distance[i] = (TimeElapsed[i] * 34300) / 2
 
    return list(distance)

# For ach file in the directory
def detect_objects(net, im):
    dim = 200
    
    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(im, 1.0, size=(dim, dim), mean=(0,0,0), swapRB=True, crop=False)

    # Pass blob to the network
    net.setInput(blob)
    
    # Peform Prediction
    objects = net.forward()
    
    
    return objects


def display_text(im, text, x, y):
    
    # Get text size 
    textSize = cv2.getTextSize(text, FONTFACE, FONT_SCALE, THICKNESS)
    dim = textSize[0]
    baseline = textSize[1]
            
    # Use text size to create a black rectangle    
    cv2.rectangle(im, (x,y-dim[1] - baseline), (x + dim[0], y + baseline), (0,0,0), cv2.FILLED);
    # Display text inside the rectangle
    cv2.putText(im, text, (x, y-5 ), FONTFACE, FONT_SCALE, (0, 255, 255), THICKNESS, cv2.LINE_AA)
    #print("Objects:",text)
    global string_obj
    string_obj = str(text)



def display_objects(im, objects, threshold):

    rows = im.shape[0]; cols = im.shape[1]

    # For every Detected Object
    for i in range(objects.shape[2]):
        # Find the class and confidence 
        classId = int(objects[0, 0, i, 1])
        score = float(objects[0, 0, i, 2])
        
        # Recover original cordinates from normalized coordinates
        x = int(objects[0, 0, i, 3] * cols)
        y = int(objects[0, 0, i, 4] * rows)
        w = int(objects[0, 0, i, 5] * cols - x)
        h = int(objects[0, 0, i, 6] * rows - y)
        
        # Check if the detection is of good quality
        if score > threshold:
            display_text(im, "{}".format(labels[classId]), x, y)
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), 2)
    
    # Convert Image to RGB since we are using Matplotlib for displaying image
    mp_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,10))
    plt.imshow(mp_img)
    #plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def motor_rotation(motor_direction):
    for i in range(0,1000):
        gpio.output(led,True)
        if(motor_direction == 'c'):
            #print('motor running clockwise\n')
            gpio.output(motor_channel, (gpio.HIGH,gpio.LOW,gpio.LOW,gpio.LOW))
            sleep(0.0009)
            gpio.output(motor_channel, (gpio.HIGH,gpio.HIGH,gpio.LOW,gpio.LOW))
            sleep(0.0009)
            gpio.output(motor_channel, (gpio.LOW,gpio.HIGH,gpio.LOW,gpio.LOW))
            sleep(0.0009)
            gpio.output(motor_channel, (gpio.LOW,gpio.HIGH,gpio.HIGH,gpio.LOW))
            sleep(0.0009)
            gpio.output(motor_channel, (gpio.LOW,gpio.LOW,gpio.HIGH,gpio.LOW))
            sleep(0.0009)
            gpio.output(motor_channel, (gpio.LOW,gpio.LOW,gpio.HIGH,gpio.HIGH))
            sleep(0.0009)
            gpio.output(motor_channel, (gpio.LOW,gpio.LOW,gpio.LOW,gpio.HIGH))
            sleep(0.0009)
            gpio.output(motor_channel, (gpio.HIGH,gpio.LOW,gpio.LOW,gpio.HIGH))
            sleep(0.0009)

        elif(motor_direction == 'a'):
            #print('motor running anti-clockwise\n')
            gpio.output(motor_channel_reversed, (gpio.HIGH,gpio.LOW,gpio.LOW,gpio.LOW))
            sleep(0.0009)
            gpio.output(motor_channel_reversed, (gpio.HIGH,gpio.HIGH,gpio.LOW,gpio.LOW))
            sleep(0.0009)
            gpio.output(motor_channel_reversed, (gpio.LOW,gpio.HIGH,gpio.LOW,gpio.LOW))
            sleep(0.0009)
            gpio.output(motor_channel_reversed, (gpio.LOW,gpio.HIGH,gpio.HIGH,gpio.LOW))
            sleep(0.0009)
            gpio.output(motor_channel_reversed, (gpio.LOW,gpio.LOW,gpio.HIGH,gpio.LOW))
            sleep(0.0009)
            gpio.output(motor_channel_reversed, (gpio.LOW,gpio.LOW,gpio.HIGH,gpio.HIGH))
            sleep(0.0009)
            gpio.output(motor_channel_reversed, (gpio.LOW,gpio.LOW,gpio.LOW,gpio.HIGH))
            sleep(0.0009)
            gpio.output(motor_channel_reversed, (gpio.HIGH,gpio.LOW,gpio.LOW,gpio.HIGH))
            sleep(0.0009)
        
    gpio.output(led,False)
        

try:
    while(True):
        time.sleep(0.9)
        if(gpio.input(p)==True):
            
            gpio.output(out,True)
            
            camera.resolution = (1280,720)
            camera.rotation = 180
            time.sleep(2)
            filename = "/home/pi/Object_Detection/images/img100.jpg"
            camera.capture(filename)
            print("Done")
            gpio.output(out,False)

            modelFile = "models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb"
            configFile = "models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
            classFile = "coco_class_labels.txt"

            if not os.path.isdir('models'):
                os.mkdir("models")

            if not os.path.isfile(modelFile):
                os.chdir("models")
                # Download the tensorflow Model
                urllib.request.urlretrieve('http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz', 'ssd_mobilenet_v2_coco_2018_03_29.tar.gz')

                # Uncompress the file
                get_ipython().system('tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz')

                # Delete the tar.gz file
                os.remove('ssd_mobilenet_v2_coco_2018_03_29.tar.gz')

                # Come back to the previous directory
                os.chdir("..")


            with open(classFile) as fp:
                labels = fp.read().split("\n")
            #print(labels)


            # Read the Tensorflow network
            net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
            FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
            FONT_SCALE = 0.7
            THICKNESS = 1

            im = cv2.imread('/home/pi/Object_Detection/images/img100.jpg')
            objects = detect_objects(net, im)
            
            display_objects(im, objects,0.59)
            
            print("detected object is a" ,string_obj)
            
            if((string_obj=='apple')or(string_obj=='banana')):
                motor_rotation('c')
                [dist1,dist2] = distance()
                print ("Measured Distances = {:.1f},{:.1f} cm".format(dist1,dist2))
                
            else:
                motor_rotation('a')
                [dist1,dist2] = distance()
                print ("Measured Distances = {:.1f},{:.1f} cm".format(dist1,dist2))
            
            if(dist1<10 or dist2<10):
                gpio.output(arduino,True)
            
            else:
                gpio.output(arduino,False)
                
                
except KeyboardInterrupt:
    gpio.cleanup()
    sys.exit(0)

except NameError:
    gpio.cleanup()
    sys.exit(0)
    
