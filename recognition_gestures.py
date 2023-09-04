import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

################### Loading the image #######################################

image = mpimg.imread('megan.jpg')
#plt.imshow(image)
#plt.show()

print(image.shape, image.shape[0] * image.shape[1] * 3)  #(337, 600, 3) 606600

print(type(image))   #<class 'numpy.ndarray'>

image_blob = cv2.dnn.blobFromImage(image = image, scalefactor = 1.0 / 255,
                                   size = (image.shape[1], image.shape[0]))

print(type(image_blob), image_blob.shape)  #<class 'numpy.ndarray'> (1, 3, 337, 600)


############################# Loading Caffe Network ################################

network = cv2.dnn.readNetFromCaffe('pose_deploy_linevec_faster_4_stages.prototxt', 'pose_iter_160000.caffemodel')
print(network.getLayerNames(), len(network.getLayerNames()))

########################### Predict Body Points ###################################

network.setInput(image_blob)
output = network.forward()

position_width = output.shape[3]
position_heigth = output.shape[2]

num_points = 15
points = []
threshold = 0.1
for i in range(num_points):
  confidence_map = output[0, i, :, :]
  _, confidence, _, point = cv2.minMaxLoc(confidence_map)

  x = int((image.shape[1] * point[0]) / position_width)
  y = int((image.shape[0] * point[1]) / position_heigth)
  
  if confidence > threshold:
    cv2.circle(image, (x, y), 5, (0,255,0), thickness = -1)
    cv2.putText(image, '{}'.format(i), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255))
    points.append((x,y))
  else:
    points.append(None)

plt.figure(figsize=(7,5))
plt.imshow(image)
plt.axis('off')
plt.show()

connection_points = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7],[1,14],
                     [14,8], [8,9], [9,10], [14,11], [11,12], [12,13]]

for connection in connection_points:
  partA = connection[0]
  partB = connection[1]
  if points[partA] and points[partB]:
    cv2.line(image, points[partA], points[partB], (255,0,0))

plt.figure(figsize=(7,5))
plt.imshow(image)
plt.axis('off')
plt.show()