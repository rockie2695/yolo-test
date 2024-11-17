from ultralytics import YOLO

# Load a pretrained YOLO11n-seg Segment model
model = YOLO("models/yolo11m-seg.pt")

# Run inference on an image
results = model(["files/image1.jpg", "files/image2.jpg", "files/image3.jpg"])  # results list

# View results
for idx, result in enumerate(results):
    print(result.masks)  # print the Masks object containing the detected instance masks

    result.show()  # display to screen
    result.save(filename=f"files/results-seg-{idx}.jpg", conf=True)  # save to disk
    result.save_txt(
        f"files/results-seg{idx}.txt", save_conf=True
    )  # save labels as .txt
    json_result = result.to_json(normalize=False, decimals=5)
    print(json_result)
