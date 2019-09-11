# STPGAN
Signal to Picture GAN
![STPGAN](../master/stpgan.gif)

In this repository are the codes to implement the signal to picture GAN project. The basic execution requirements are
---
**CUDA 9.0
---
CuDNN 7.4.1.5
---
Tensorflow-GPU 1.12.0
---
Tensorboard 1.12.2
---
OpenCV 4.1.1.26
---
Keras 2.2.5**
---

The order of execution of the files is first

"preprocessing.py"

then

"stpgan2.py"

to load the models and generate new samples

"cargajsonymodelos.py"

and to store the synthetic signals.

"vectorizacion.py"

Once the file has been stored it must be opened with Excel, and replace "." with "," .
