# -*- coding: utf-8 -*-
"""
Created on Sat May  2 13:27:08 2020

@author: okeyr
"""

#!/usr/bin/python3
# write tkinter as Tkinter to be Python 2.x compatible
from tkinter import *
import pyaudio
import wave
from select import select
import winsound
import datetime,time
import sys

import threading 
import os 

def startButton(event):
    t_met = threading.Thread(target = metronome.start) 
    t_met.start() 

class Recorder(object):
    def __init__(self,chunk,sample_format,channels,
                  fs,filename,bpm):
        self.chunk=chunk
        self.sample_format=sample_format
        self.channels=channels
        self.fs=fs
        self.filename=filename+".wav"
        self.recording=False
        self.frames=[]
        self.p = pyaudio.PyAudio()  # Create an interface to PortAudio
        self.stream = self.p.open(format=self.sample_format,
                channels=self.channels,
                rate=self.fs,
                frames_per_buffer=self.chunk,
                input=True)
        
    def start(self):
        self.recording = True
        time.sleep(metronome.first_call-time.time())
        while(self.recording):
            # self.start_recording=time.time()
            data = self.stream.read(self.chunk)
            self.frames.append(data)
            print("...recording...")
            
        print("Recording stopped")
    
    def stop(self):
        self.recording=False
        t_met_stop = threading.Thread(target = metronome.stop) 
        t_met_stop.start()
        # Stop and close the stream 
        self.stream.stop_stream()
        self.stream.close()
        # Terminate the PortAudio interface
        self.p.terminate()
        self.save_file()
    
    def save_file(self):
        # Save the recorded data as a WAV file
        print(metronome.first_call)
        # phase_samples = round((self.start_recording - metronome.first_call)*self.fs/1000)
        # frames_in_2bar = round(self.fs/1000*2*metronome.barTime)
        # frames = self.frames[phase_samples:phase_samples+frames_in_2bar]
        # print(len(frames))
        wf = wave.open(self.filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.sample_format))
        wf.setframerate(self.fs)
        wf.writeframes(b''.join(self.frames))
        wf.close() 
        print("File saved")
        sys.exit()
        
class Metronome(object):
    def __init__(self,bpm):
        self.bpm = bpm
        self.beatTime = 60/bpm
        self.barTime = 4*self.beatTime
        self.barSoundFreq = 1500
        self.beatSoundFreq = 1000
        self.soundTime = 100
        self.metronome = True
        
    def start(self):
        next_call = time.time()
        self.first_call=next_call+self.barTime
        t_rec = threading.Thread(target = recorder.start) 
        t_rec.start()
        count = 0
        while (self.metronome):
            print(datetime.datetime.now())
            if count%4==0:
                winsound.Beep(self.barSoundFreq, self.soundTime) 
            else:
                winsound.Beep(self.beatSoundFreq, self.soundTime)
            count = count + 1;
            next_call = next_call+self.beatTime;
            time.sleep(next_call - time.time())
    
    def stop(self,event):
        self.metronome=False
        # t_rec_stop = threading.Thread(target = recorder.stop()) 
        # t_rec_stop.start()
        recorder.stop()

recorder = Recorder(1024, pyaudio.paInt16,1,44100,"output",100)
metronome = Metronome(100)

buttonStart = Button(None, text='Start recording')
buttonStop = Button(None, text='Stop recording')
buttonStart.pack()
buttonStop.pack()
buttonStart.bind('<Button-1>', startButton)
buttonStop.bind('<Button-1>', metronome.stop)
buttonStart.mainloop()
buttonStop.mainloop()

    
        


