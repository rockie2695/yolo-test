from ultralytics import YOLO

# Load a model
model = YOLO("models/yolo11m-pose.pt")  # load an official model

# Predict with the model
results = model(
    ["files/image3.jpg"]
)  # predict on an image
for idx, result in enumerate(results):
    print(result.keypoints)  # Keypoints object for pose outputs

    result.show()  # display to screen
    result.save(filename=f"files/results-pose-{idx}.jpg", conf=True)  # save to disk
    result.save_txt(
        f"files/results-pose{idx}.txt", save_conf=True
    )  # save labels as .txt
    json_result = result.to_json(normalize=False, decimals=5)
    print(json_result)
