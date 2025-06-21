"""DOCUMENTATION

Requirements:
    The modules that are required for this experiment are the following:
        1. pyaudio
        2. wave
        3. random
        4. cv2
        5. os 
        6. pathlib
        7. numpy
        7. matplotlib.pyplot 
        8. soundfile
        9. sys
        10. time
        
Objective:
1.Define a system to add two audio signals(live audio recording/ audio available in the system) 
and another system to change playback rate
2.Analyse property: Linearity and Time Varience for the playback rate system
3.Analyse the system in frequency domain

Process: 
1) This code provides the user live audio recording feature or choose from samples provide.This
    code uses pyaudio to record new audio and save it as .wav file.
2) Uses soundfile module to read the two audio signals with sample rate sr1,sr2 .
3) Uses numpy for averaging the channels if audio signal r1 or r2 have more than one. np.mean()
    is used for the conversion
4) matplotlib.pyplot module is used for plotting the two audio signals and combined signal(r1+r2) 
5) Provides the user an option to perfom operations like time scale,linearity,time-invaraince,
   dtft according to the choice given.("1.Scale any Audio\n2.Fourier Tranform\n3.Linearity 
   and Time Invarience")
6) Uses custom made function to compute the DTFT of the audio files(but slower compared to FFT)
7) Uses cv2 module for opening the webcam(cv2.Videocapture()) and provides live feed. Press "s"
   for capturing the image and save it in jpg format. Press "q" to exit camera.
8) Uses custom made funtions for resizing the audio signals,dtft,time-invariance,linearity etc.
"""





#Importing required modules

import pyaudio  # For capturing live audio
import wave   # For reading and writing .wav files
import random  # For generating random numbers 
import cv2    # For handling webcam input and image capture
import os     # For interacting with the filesystem
from pathlib import Path # For handling file paths
import numpy as np  # For performing mathematical operations
import matplotlib.pyplot as plt # For plotting audio signals
import soundfile as sf  # To read and write audio files
import sys  # System exit handling

def Home():
    print("WELCOME TO SIGNAL PROCESSING PROJECT")
    
    print("What operation do you need to perform :\n")
    print("\t\t1.Audio Signal Processing\n\t\t2.Image processing")  
    # press 1 for audio signal processing
    #press 2 for image processing
    ch=int(input("Enter your choice :"))
    if(ch==1):
        s=[0,0]
        for i in range(2):
            print("What audio (",i+1,") do you need to process :")
            print("\t\t1.Record Audio\n\t\t2.Sample Audio")  
            ch1=int(input("Enter your choice :"))  
            #press 1 for live audio record and Press 2 for sample audio available
            if(ch1==1):       
                s[i]=record_audio()         #provide a live audio for audio signal processing
            elif(ch1==2):
                print("The Available music:")
                s[i]=choosemusic()         #choose available music from the system
            else:
                print("Invalid Choice")
            
        r1,sr1= sf.read(s[0])               #Using soundfile reading the audio
        if len(r1.shape) > 1:              
            r1 = np.mean(r1, axis=1)
        t1 = np.linspace(0, len(r1) / sr1, num=len(r1))
        r2,sr2= sf.read(s[1])
        
        if len(r2.shape) > 1:
            r2 = np.mean(r2, axis=1)
        t2 = np.linspace(0, len(r2) / sr2, num=len(r2))
        
        
        #plotting the orginal signal and combined audio signal
        plt.subplot(311)
        plt.title("Audio Signal-1")
        plt.plot(t1,r1)                 #audio 1
        plt.grid()
        plt.subplot(312)
        plt.title("Audio Signal-2")
        plt.plot(t2,r2)                 #audio 2
        plt.grid()
        r11,r22=resize(r1,r2)       #resizing the signals r1 and r2
        r3=r11+r22
        t3= np.linspace(0, len(r3) / sr2, num=len(r3))
        combined_output = "/home/hariprasad/Downloads/combined.wav"
        sf.write(combined_output,r3, sr1)  #write into a new file
        print(f"Stretched audio saved as {combined_output}")
        plt.subplot(313)
        plt.plot(t3,r3)                  #combined audio(audio 1 + audio 2)
        plt.title("Combined Audio")
        plt.grid()
        plt.tight_layout()
        plt.show()
        
        #menu driven program for performing operations on the signals
        print("What opertion do you need to perform:")
        print("1.Scale any Audio\n2.Fourier Tranform\n3.Linearity and Time Invarience")
        #press 1 for time scaling
        #press 2 for fourier transform(dtft)
        #press 3 for linearity and time invarience check
        ch2=int(input("Enter your Choice(1/2/3):"))
        if(ch2==1):
            suf = float(input("Enter the playback rate of the audio: "))
            # suf<1.0 -->slows down the audio(doubles the duration of audio)
            # suf>1.0 -->speeds up the  audio(halves the duration audio)
            y_stretched = assignplayrate(r1, suf)           #streching the audio
            stretched_output = "/home/hariprasad/Downloads/stretched.wav"
            sf.write(stretched_output, y_stretched, sr1)   #writing stretched audio into new file
            print(f"Stretched audio saved as {stretched_output}")
            plt.subplot(211)
            plt.title("Original signal")
            plt.plot(t1,r1)
            plt.grid()
            plt.subplot(212)
            plt.title("Stretched signal")
            plt.plot(np.linspace(0,len(y_stretched)/sr1,len(y_stretched)),y_stretched)   #stretched audio
            plt.grid()
            plt.tight_layout()
            plt.show()   
        elif(ch2==2):                        #dtft
            w=np.linspace(-np.pi,np.pi,1000)    # w value between -pi to +pi with 1000 samples
            dtft(w,r1)
            dtft(w,r2)
        
        elif(ch2==3):
            print("stretched waveform is ",printing(a1),"and",z2)
    elif(ch==2):
        print("press 's' to capture press ,'q' to quit\nTuring on the camera pls wait...")
        cam()



def choosemusic():
    directory =Path("/home/hariprasad/Music/")  # Define the directory containing music files
    files = [f.name for f in directory.iterdir() if f.is_file()]
    for i in range(1,len(files)+1):
        print(i,".",files[i-1])
    s1=int(input("Enter the Song :"))
    s2="/home/hariprasad/Music/"+files[s1-1] # Constructing the full path of the selected song
    return s2

def assignplayrate(y, playrate):  # Function to assign the play rate
    new_length = int(len(y) / playrate)  # Calculate the new length
    indices = np.linspace(0, len(y) - 1, new_length).astype(int)  # Adjust the sample rate
    return y[indices]  # Select only valid indices


def record_audio():
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100  # Sampling rate in Hz
    CHUNK = 1024  # Buffer size
    DURATION = int(input("Enter Recording duration in seconds(max:10s):")) # Recording duration in seconds
    OUTPUT_FILENAME = "recorded_audio"+str(random.randint(0,1000))+".wav"  # Saving the audio file in .wav format
 
    """Records audio for a set duration and saves it to a file."""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, 
                        input=True, frames_per_buffer=CHUNK)
    
    print("Recording...")
    frames = [stream.read(CHUNK) for _ in range(0, int(RATE / CHUNK * DURATION))]
    print("Recording finished.")
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # Save to file
    with wave.open(OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    print(f"Audio saved as {OUTPUT_FILENAME}")
    return OUTPUT_FILENAME


def cam():
    
    cap = cv2.VideoCapture(0)  # Open webcam (1 = external webcam, change to 0 for default webcam)
    
    # Checking if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit()

    while True:
        
        ret, frame = cap.read()  # Capture frame-by-frame
        
        # If frame capture fails, exit loop
        if not ret:
            print("Failed to capture image.")
            break
        # Display the captured frame
        cv2.imshow("Webcam - Original", frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        # Save the frame and exit if 's' is pressed
        if key == ord('s'):
            cv2.imwrite("captured_image.jpg", frame)
            print("Image saved as captured_image.jpg")
            break      
        
        elif key == ord('q'):   # Exit if 'q' is pressed
            print("Quitting...")
            break
    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    
def resize(y1,y2):
    if(len(y1)>len(y2)):
        dif=len(y1)-len(y2)
        y22=np.append(y2,np.zeros(dif))#Equalising the size
        return y1,y22
    
    
    elif(len(y2)>len(y1)):
        dif=len(y2)-len(y1)
        y11=np.append(y1,np.zeros(dif))#Equalising the size
        return y11,y2
    else:
        return y1,y2

def dtft(w, x):
    k = np.zeros(len(w), dtype=complex)
    n_vals = np.arange(len(x))
    for i in range(len(w)):
        k[i] = np.sum(x * np.exp(-1j * w[i] * n_vals))  # Analysis equation for determing fourier transfrom
    plt.subplot(211)
    plt.title("Magnitude spectrum")
    plt.plot(w,np.abs(k))           # Magnitude spectrum
    plt.grid()
    plt.subplot(212)
    plt.title("Phase Spectrum")
    plt.plot(w,np.angle(k))         # Phase spectrum
    plt.grid()
    plt.show()

#Defining two arrays for properties check    
x1=np.array([1,2,3,4,5],dtype="float32")
x2=np.array([6,7,8,9,10],dtype="float32")
c=np.array(x1+x2)
a1=[]

#LINEARITY CHECK

#Step1 = Superposition check
def su(s,a,b):
    y=[]
    y1=[]
    y2=[]
    y3=[]
    y.append(s(a,2))      
    y1.append(s(b,2))
    y2.append(s(c,2))
    for i in range(len(y)):
        y3.append(y[i]+y1[i])
    if(np.allclose(y2,y3)==True):
        return "superpos"
    else:
        return "nonsuper pos"
 
a1.append(su(assignplayrate,x1,x2))     

#Step2 = Homogenity check
def homo(s,a):
    y=[]
    y1=[]
    y=s(2*a,2)
    y1=(2*np.array(s(a,2)))
    if(np.allclose(y,y1)==True):
        return "homo"
    else:
        return "non homo"

a1.append(homo(assignplayrate,x1))
def printing(a):    
    if a[0]=="superpos" and a[1]=="homo": 
        return "linear"
    else:
        return "non linear"
    
#TIME INVARIANCE CHECK

def inv(s,a,c):
    a2=[]
    b2=[]
    a3=np.roll(a,c)
    a2.append(s(a3,2))    #y(t-to)
    b2.append(s(a,2))     #x(t-to)
    b3=np.roll(b2,c)
    if np.allclose(a2,b3):         #Comparing both equations to check for time invariance
        return "time invariant"
    else:
        return "time variant"
z2=inv(assignplayrate,x1,2)
c="y"
while(c=="y"):
    Home()
    # Asking the user whether you want to contiue
    c=input("Do you want to continue (y/n):")
    # Press y to continue and Press n to quit
    while (c!="y" and c!="n"):
        c=input("Do you want to continue (y/n):")

"""
Contributors

Name:                     Roll no:
Christin Prakash          CB.EN.U4ECE24208
Dinesh Kuthalanathan T    CB.EN.U4ECE24211
Gokul S                   CB.EN.U4ECE24212
Hariprasad G              CB.EN.U4ECE24213


"""