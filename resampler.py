# Bryce McIntyre

import scipy
import numpy as np
import librosa as lib
import matplotlib.pyplot as plt
#import yodel as yod

data = []

for i in range(1,22):
    filename = "/Users/marksinks/Documents/CS701a/recordings/Other/Other/random" + str(i)+ ".wav"
    y, sr = lib.load(filename, sr = 44100)
    lib.output.write_wav("/Users/marksinks/Documents/CS701a/recordings/Other/" + "random"+ str(i)+".wav",y, sr)
    #S = lib.feature.melspectrogram(y=y, sr=sr)
    #np.savetxt("melspecs/melspec"+str(i)".csv" , S, fmt='% f', delimiter=",")

#s = scipy.signal.wiener(S)
