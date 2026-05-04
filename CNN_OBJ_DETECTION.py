#%pip install opencv-python
import cv2

# Load pre-trained model (MobileNet SSD)
net = cv2.dnn.readNetFromCaffe(
    "MobileNetSSD_deploy.prototxt",
    "MobileNetSSD_deploy.caffemodel"
)

# Class labels
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Load image
image = cv2.imread("image.jpg")
(h, w) = image.shape[:2]

# Preprocess image
blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)

# Pass through network
net.setInput(blob)
detections = net.forward()

# Loop over detections
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > 0.5:
        idx = int(detections[0, 0, i, 1])

        box = detections[0, 0, i, 3:7] * [w, h, w, h]
        (startX, startY, endX, endY) = box.astype("int")

        label = CLASSES[idx]

        # Draw bounding box
        cv2.rectangle(image, (startX, startY), (endX, endY), (0,255,0), 2)
        cv2.putText(image, label, (startX, startY-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

# Show result
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Download Model Files: The error indicated missing model files: MobileNetSSD_deploy.prototxt and MobileNetSSD_deploy.caffemodel. You need to download these files. They are commonly available from the official OpenCV GitHub repository or examples. For instance, you can search for 'MobileNet-SSD Caffe model files download' to find reliable sources. Download both files and place them into your project folder alongside object_detection.py.