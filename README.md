# Real-time gesture-recognition by 2D input pose

With the rapid development of deep learning algorithms and high-level calculation devices, the intelligent system can be in charge of more complicated and challengeable missions for human activity recognition with higher accuracy than before. Industrial partners have already showed their great interests in the implementation of AI in real-life applications, such as autonomous driving, human-robot collaboration and healthcare monitoring. In order to fit the growing industrial needs, more efficient and lower-latency gesture recognition methods should be developed to solve the real-time problems in those intelligent systems.  

This master project is based on the idea of input 2D skeletal body joints for an LSTM RNN Network to complete the human gesture recognition in real time. Our assumption is: 

1 Deep Learning algorithms (CNN, RNN) could help us to solve the real-time problems in gesture recognition without a drop in accuracy compared to the traditional machine learning methods (random forest, SVM, Logistic Regression, ...) 

2 Using a series of 2D poses, rather than 3D poses or a raw 2D images, can produce an accurate estimation of the behavior of a person 

3 Combining the pre-trained network 'openpose' (https://github.com/CMU-Perceptual-Computing-Lab/openpose's) and self-trained RNN could make the gesture recognition in real-time with high accuracy and low latency, compared to the existing posture recognition methods, or other offline gesture recognition methods. However, we also expect this method would exceed the state-of-the-art performance (like 3D CNN， TSN) in real-time and accuracy   
