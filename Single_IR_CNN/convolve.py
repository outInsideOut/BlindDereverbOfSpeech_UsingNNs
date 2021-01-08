from scipy import signal
import soundfile as sf
import numpy as np
import librosa
import os

# read in IR as array
IR, fs = librosa.load('CarpenterCenter.wav', sr=8000)
IR = np.asarray(IR)

for filename in os.listdir():
    # if an audio file
    if filename.endswith(".flac") and not filename.startswith('v_') and not filename.startswith('c_'):
        # load into array 
        clean, fs = librosa.load(filename, sr=8000)
        # convolve with IR
        filtered = signal.convolve(clean, IR, mode='full')
        # if too short, pad
        if len(clean) < 40000:
            silence = 40000 - len(clean)
            np.pad(clean, silence)
        if len(filtered) < 40000:
            silence = 40000 - len(filtered)
            np.pad(filtered, silence)
        # cut down to 5 seconds
        clean = clean[0:40000,]
        filtered = filtered[0:40000,] 

        # format names
        cleanName = "c_{}".format(filename)
        filteredName = "v_{}".format(filename)
        # write files
        sf.write(filteredName, filtered, 8000)
        sf.write(cleanName, clean, 8000)
    else:
        continue
