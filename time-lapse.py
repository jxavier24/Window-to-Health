import time
import subprocess
from time import sleep
import picamera
import RPi.GPIO as GPIO

GPIO.setmode(GPIC.BCM)

buzzTime = 1
buzzDelay = 2
buzzerPin = 4
GPIO.setup(buzzerPin, GPIO.OUT)
    
#Obtain temp readings from thermal sensor here:
    #temp_1 = Sensor.temperature

WAIT_TIME = 3600
with picamera.PiCamera() as camera:
    #Set camera resolution
    camera.resolution = (1024,768)
    #Take picture
    for filename in camera.capture_continuous('/home/pi/time-lapse/img{timestamp:%h-%M-%S-%f}.jpg')
        #Send picture to Dropbox
        from subprocess import call
        photofile = "/home/pi/Dropbox-Uploader/dropbox_uploader.sh upload" + filename
        call ([photofile], shell=True)
        #Buzzer alerts patient if temperature is above certain value
        if (temp_1 >= 100):
            GPIO.output(buzzerPin, True)
            sleep(buzzTime)
            GPIO.output(buzzerPin, False)
            sleep(buzzDelay)
        sleep(WAIT_TIME)