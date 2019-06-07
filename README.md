# Object-Detection-using-Pytorch

### Before addressing Object detection we need to have an idea of Convolution Neural networks(CNN)

### *The process goes like this*

- here I am using pre-trained SSD (Single Shot multibox Detector)
- Video will be broken down into frames 
- SSD will be run on each frame trying to detect an object
- If the score of the returned detection tensor has a score of more than 0.6 then the object has been detected
- Return the frame with the object detected



![SSD](/images/ssd.png)
