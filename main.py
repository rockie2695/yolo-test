from ultralytics import YOLO
import cv2

# Load a model
model = YOLO("yolo11n.pt")  # pretrained YOLO11n model

# Run batched inference on a list of images
results = model(["image1.jpg", "image2.jpg"])  # return a list of Results objects
# results = model(["image1.jpg", "image2.jpg"], stream=True)  # return a generator of Results objects
# results=model("screen") # for screen capture
# results=model("https://ultralytics.com/images/bus.jpg") # for online images
# results=model("video.mp4", stream=True) # for video file
# results=model("path/to/dir", stream=True) # Define path to directory containing images and videos for inference
# results=model("https://youtu.be/LNwODJXcvt4", stream=True) # for video file
# model.predict("bus.jpg", save=True, imgsz=320, conf=0.5) #there is more config
# Process results list
for idx, result in enumerate(results):
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    print(boxes, masks, keypoints, probs, obb)

    result.show()  # display to screen
    result.save(filename=f"results{idx}.jpg")  # save to disk

# test 2
results2 = model("https://youtu.be/nZF0poe3Pus", stream=True)

# Process results generator
for idx, result in enumerate(results2):
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    print(boxes, masks, keypoints, probs, obb)

    result.show()  # display to screen
    result.save(filename=f"video_results/results{idx}.jpg")
    result.save_crop(save_dir="video_results", file_name=f"results{idx}.jpg")
