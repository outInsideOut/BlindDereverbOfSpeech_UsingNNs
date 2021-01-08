# Blind Dereverberation of Speech using Transformative GANs

## Description

This repo is the housing for A degree level dissertation-style final-year project for my BSc in 
Computer Science from the University of Lincoln.

This will entail creating simple CNN and GANs tailored towards the speech enhancement problem, 
then trying to find an optimum combination of alternating loss functions / learning patterns to maximise 
the signal:noise ratio in respect to speech:reverb.

## Data

The data being used is the <a href="http://www.openslr.org/12">LibreSpeech Corpus</a>,
A collection of (English) speech recordings which is available for public use within education.

This data is then convolved using the <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html">scipy.signal.convolve</a>
function and open-source Impulse Response Convolutions from the <a href="echotheif.com">EchoThief</a> library.
The raw data can be used as a taget to learn towards in a traditional feed-forward CNN or as an example of a 
legitimate solution for a GAN-discriminator.


