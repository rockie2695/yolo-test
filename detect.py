from ultralytics import YOLO
import cv2

# Load a model
model = YOLO("models/yolo11m.pt")  # pretrained YOLO11n model

# Run batched inference on a list of images
results = model(
    ["files/image1.jpg", "files/image2.jpg", "files/image3.jpg"]
)  # return a list of Results objects
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
    print(
        "boxes",
        boxes,
    )

    result.show()  # display to screen
    result.save(filename=f"files/results{idx}.jpg", conf=True)  # save to disk
    result.save_txt(f"files/results{idx}.txt", save_conf=True)  # save labels as .txt
    json_result = result.to_json(normalize=False, decimals=5)
    print(json_result)

# # test 2
# results2 = model("https://youtu.be/nZF0poe3Pus", stream=True)

# # Process results generator
# for idx, result in enumerate(results2):
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     print(boxes, masks, keypoints, probs, obb)

#     result.show()  # display to screen
#     result.save(filename=f"video_results/results{idx}.jpg")
#     result.save_crop(save_dir="video_results", file_name=f"results{idx}.jpg")

# test 3
# Open the video file
# video_path = "files/test_1080_1920_30fps.mp4"
# cap = cv2.VideoCapture(video_path)

# # Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()

#     if success:
#         # Run YOLO inference on the frame
#         results = model(frame)

#         # Visualize the results on the frame
#         annotated_frame = results[0].plot()

#         # Display the annotated frame
#         cv2.imshow("YOLO Inference", annotated_frame)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         # Break the loop if the end of the video is reached
#         break

# # Release the video capture object and close the display window
# cap.release()
# cv2.destroyAllWindows()
