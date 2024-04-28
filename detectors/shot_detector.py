import cv2
import pickle
from ultralytics import YOLO  # Assuming YOLO module exists

class ShotDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        shot_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                shot_detections = pickle.load(f)
            return shot_detections

        for frame in frames:
            shot_dict = self.detect_frame(frame)
            shot_detections.append(shot_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(shot_detections, f)
        
        return shot_detections

    def detect_frame(self, frame):
        # Assuming model returns a dictionary containing bounding boxes and class labels
        results = self.model.predict(frame)
        
        return results

    def draw_bboxes(self, video_frames, shot_detections, previous_detections=None):
        output_video_frames = []
        for frame, detection in zip(video_frames, shot_detections):
            for bbox in detection['boxes']:
                x1, y1, x2, y2 = bbox.xyxy.tolist()[0]
                class_name = bbox.cls.tolist()[0]

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, class_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            output_video_frames.append(frame)
        
        return output_video_frames
