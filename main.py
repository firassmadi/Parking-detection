from pathlib import Path
from ultralytics import YOLO
import cv2

MEDIA_DIR = Path('.') / ''
media_path = MEDIA_DIR / 'demo.mp4'  # Change to the name of your media file (can be .mp4 or an image file like .jpg)
media_path_out = media_path.with_stem(media_path.stem + '_out').with_suffix('.mp4' if media_path.suffix == '.mp4' else media_path.suffix)

model_path = Path('.') / 'runs' / 'detect' / 'train' / 'weights' / 'last.pt'

# Load a model
model = YOLO('best.pt')  # load a custom model

threshold = 0.5

def process_frame(frame):
    results = model(frame)[0]
    total_spots = 0
    available_spots = 0

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            class_name = results.names[int(class_id)].upper()
            color = (0, 255, 0) if class_name == "BOS" else (0, 0, 255) if class_name == "DOLU" else (255, 255, 255)
            if class_name == "BOS":
                available_spots += 1
            total_spots += 1
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
            cv2.putText(frame, class_name, (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, color, 2, cv2.LINE_AA)

    info_text = f"Available Spots: {available_spots}/{total_spots}"
    
    # Determine text size and position
    (text_width, text_height), baseline = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, 2)
    x, y = 10, 30
    cv2.rectangle(frame, (x, y - text_height - baseline), (x + text_width, y + baseline), (0, 0, 0), cv2.FILLED)
    
    # Add text to the frame
    cv2.putText(frame, info_text, (x, y),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

if media_path.suffix in ['.mp4', '.avi', '.mov']:  # Video files
    cap = cv2.VideoCapture(str(media_path))
    ret, frame = cap.read()
    H, W, _ = frame.shape
    out = cv2.VideoWriter(str(media_path_out), cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

    while ret:
        frame = process_frame(frame)
        out.write(frame)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break
        ret, frame = cap.read()

    cap.release()
    out.release()
    cv2.destroyAllWindows()

elif media_path.suffix in ['.jpg', '.jpeg', '.png', '.bmp']:  # Image files
    frame = cv2.imread(str(media_path))
    frame = process_frame(frame)
    cv2.imwrite(str(media_path_out), frame)
    cv2.imshow('Image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Unsupported media format.")
