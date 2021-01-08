# Single IR CNN for speech de-reverb

## Description

This Convolutional Neural Network is implemented to show a proof of concept of a simple speech enhancement network 
which aims to increase the signal:noise ratio of the provided data with respect to speech:reverb.

Reverb has been added to samples from the <a href="http://www.openslr.org/12">librespeech corpus</a> using 
a single impulse response convultion from the <a href="http://www.echothief.com/">echo thief collection</a>. 
This was applied to the audio using the
<a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html">scipy.signal.convolve()</a>
function.
This is executed using the convolve.py script within this repo.

