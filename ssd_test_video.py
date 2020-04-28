import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='3'

from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import matplotlib.pyplot as plt
import cv2
import sys
import time
import numpy as np

timer = Timer()

# # MobileNet SSD #
# net_type = "mb1-ssd"
# model_path = "models/mobilenet-v1-ssd-mp-0_675.pth"
# label_path = "models/voc-model-labels.txt"
# # ------------- #

# MobileNet SSD # --- Le Anh trained ---
# net_type = "mb1-ssd"
net_type = "mb2-ssd-lite"

# model_path = "models/mb1-ssd-Epoch-19-Loss-6.089628639675322.pth"
# model_path = "models/mb1-ssd-Epoch-105-Loss-inf.pth"
model_path = "models/mb2-ssd-lite-Epoch-35-Loss-inf.pth"

label_path = "models/voc-model-labels.txt"
# --------------- #

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
if net_type == 'vgg16-ssd':
	net = create_vgg_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd':
	net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd-lite':
	net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb2-ssd-lite':
	net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
elif net_type == 'sq-ssd-lite':
	net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
else:
	print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
	sys.exit(1)
	
net.load(model_path)

if net_type == 'vgg16-ssd':
	predictor = create_vgg_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd':
	predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd-lite':
	predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'mb2-ssd-lite':
	predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'sq-ssd-lite':
	predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
else:
	print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
	sys.exit(1)

print("Loading Trained Model is Done!\n")

# Start Detection #
print("Starting Detection...\n")

# Load video
cap = cv2.VideoCapture('videos/bdd-videos-sample.mov')
ret, one_image = cap.read()
print("Input Shape: ", one_image.shape)

# Video configuration
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('videos/mb2_35_epochs_30.avi', fourcc, 30.0, (one_image.shape[1], one_image.shape[0]))
frame_cnt = 0

# Check if video opened successfully
if (cap.isOpened()== False): print("Error opening video stream or file")

color = np.random.uniform(0, 255, size = (10, 3))

# Read until video is completed
while(cap.isOpened()):
	ret, orig_image = cap.read()

	if ret == True:

		image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
		frame_cnt += 1

		timer.start()
		boxes, labels, probs = predictor.predict(image, 10, 0.4)
		interval = timer.end()
		print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)), "Frame: ", frame_cnt)

		fps = 1/interval

		for i in range(boxes.size(0)):
			box = boxes[i, :]
			label = f"{class_names[labels[i]]}: {probs[i]:.2f}"

			i_color = int(labels[i])

			cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), color[i_color], 2)

			cv2.putText(orig_image, label,
						(box[0] - 10, box[1] - 10),
						cv2.FONT_HERSHEY_SIMPLEX,
						1,  # font scale
						color[i_color],
						2)  # line type

			cv2.putText(orig_image, "FPS = " + str(int(fps)),
				(1080, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

		# Write the flipped frame
		out.write(orig_image)

	else: break

cap.release()
out.release()

print("Check the result!")