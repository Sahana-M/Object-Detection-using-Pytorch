import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

#frame - real image, net - pretrained neural network, transform - 

def detect(frame, net, transform):
    height, width = frame.shape[:2] #getting height and width of frame
    frame_t = transform(frame)[0]   #transforming frame into numpy array    
    x = torch.from_numpy(frame_t).permute(2,0,1)#rbg to grb
    x = Variable(x.unsqueeze(0)) #unsqueeze to create a fake dimension, here 1 dimension, Variable to convert it to a torch tensor which has gradient and tensor
    y = net(x)  #output from neural network, contains torch tensor and gradient
    detections = y.data  #from this we get detections - torch tensor from y
    scale = torch.Tensor([width, height, width, height]) #inorder to normalise between 0 & 1, we need a 4x4 tensor matrix which corrsponds to all 4 corners of the detecting rectangle  
    #detection tensor contains 4 elements
    #1st element - batch(outputs of batch inputs), 2nd element-number of class-means number of objects detected in input image
    #3rd element - number of occurence of a class -no. of occurence of a particular object
    #4th element - [score, x0,y0, x1,y1] if score > 0.6 object found, if score < 0.6 object not found(means won't detect it)
    for i in range(detections.size(1)):    
        j = 0
        while detections[0,i,j,0] >= 0.6:            
            pt = (detections[0, i, j, 1:] * scale).numpy()
            cv2.rectangle(frame,  (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2) 
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            j += 1
    return frame

#create SSD neural network
net = build_ssd('test')
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage))
            
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))



reader = imageio.get_reader('epic-horses.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('output.mp4', fps = fps)

for i, frame in enumerate(reader):
    frame = detect(frame, net.eval(), transform)
    writer.append_data(frame)
    print(i)
writer.close()
    





















