import matplotlib
import matplotlib.pyplot as plt
import cv2
import os
import torch
import torchvision
import numpy as np
from IPython.display import display
from torchvision.transforms import functional as F
from PIL import Image
from mpl_toolkits.mplot3d.axes3d import Axes3D
from lib.visualize_groundtruth import create_index, viz2d, plt_imshow, bgr2rgb, plt_3dplot, coco_to_camma_kps, progress_bar
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
matplotlib.use('TkAgg')
USE_GPU = True

if torch.cuda.is_available() and USE_GPU:
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Using device:", DEVICE)

# load the model and pre-trained weights
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
# put the model on the eval mode and assign either "cpu" or "gpu" device
model.eval()
model = model.to(DEVICE)

def maskRCNN_model(img):

    # img = Image.open("./test.png")
    # convert input multi-view images to pytorch tensors
    img_tensor = [F.to_tensor(img.convert('RGB')).to(DEVICE)]

    # run the model on the multiview image
    with torch.no_grad():
        predictions_2d = model(img_tensor)
    # get the bounding boxes and keypoint detections
    boxes = predictions_2d[0]["boxes"].cpu().numpy()
    keypoints = predictions_2d[0]["keypoints"].cpu().numpy()
    # put the results in the list for the visualization
    anns = []
    for index, (b, kp) in enumerate(zip(boxes, keypoints)):
        b = [b[0], b[1], b[2] - b[0], b[3] - b[1]]
        anns.append({"bbox": b, "keypoints": coco_to_camma_kps(kp), "person_id": index, "only_bbox": 0})

    im = np.asanyarray(img)
    imgs_render_pred = viz2d(im, anns)
    # fig = plt.figure(figsize=(20, 18))
    # print("Visualizing the 2D predictions from the Keypoint-MaskRCNN model")
    # plt.imshow(bgr2rgb(imgs_render_pred))
    # plt.title("show")
    # plt.axis("off")
    # plt.show()
    cv2.namedWindow("image_result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image_result', 800, 1100)
    cv2.imshow("image_result", imgs_render_pred)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    try:
        # cv2.imwrite(os.path.join("./", "test.png"), frame)
        img = Image.fromarray(frame)
        maskRCNN_model(img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    except:
        pass

cap.release()
cv2.destroyAllWindows()
