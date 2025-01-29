from ultralytics import YOLO

model = YOLO("models\\trainrun3_150epoch_best.pt")

# Inference
results = model.predict("C:\\Users\\Pat\\Downloads\\DL\\soccer_analysis\\input_videos\\08fd33_4.mp4", save=True)

print(results[0])

print('----------------------')

for box in results[0].boxes:
    print(box)