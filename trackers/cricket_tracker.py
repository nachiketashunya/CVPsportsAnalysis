from ultralytics import YOLO 
import cv2
import pickle
import sys

class CricketTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        return player_detections

    def detect_frame(self,frame):
        results = self.model.track(frame, persist=True, show=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            if box.id != None:
                track_id = int(box.id.tolist()[0])
                result = box.xyxy.tolist()[0]
                object_cls_id = box.cls.tolist()[0]
                object_cls_name = id_name_dict[object_cls_id]
                if object_cls_name in ['batsman', 'bowler', 'non-striker', 'umpire', 'wicket-keeper']:
                    player_dict[track_id] = [result, object_cls_name]
        
        return player_dict

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            for track_id, res in player_dict.items():
                x1, y1, x2, y2 = res[0]
                player_head_center = (int((x1 + x2) / 2), int(y1))

                # Define box properties
                box_color = (255, 0, 0)  # Blue box
                border_color = (0, 0, 255)  # White border
                text_color = (255, 255, 255)  # White text
                shadow_color = (0, 0, 0)  # Black shadow
                thickness = 2
                font_scale = 0.7

                # Calculate text size
                text = f"{res[1]}"
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                # Define box dimensions with padding around the text
                text_padding = 5
                box_width = text_size[0] + 2 * text_padding
                box_height = text_size[1] + 2 * text_padding

                # Calculate top-left corner of the box relative to player head center
                box_top_left = (player_head_center[0] - int(box_width / 2), player_head_center[1] - text_size[1] - text_padding)

                # Draw the filled rectangle box
                cv2.rectangle(frame, box_top_left, (box_top_left[0] + box_width, box_top_left[1] + box_height), box_color, cv2.FILLED)

                # Draw the text with white color and a black shadow for better readability
                text_org = (box_top_left[0] + text_padding, box_top_left[1] + box_height - text_padding)
                cv2.putText(frame, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, shadow_color, thickness)
                cv2.putText(frame, text, (text_org[0] - 1, text_org[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

                # Draw the white border around the box
                cv2.rectangle(frame, box_top_left, (box_top_left[0] + box_width, box_top_left[1] + box_height), border_color, thickness)

            output_video_frames.append(frame)
                
        return output_video_frames
            