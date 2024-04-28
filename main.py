from utils import read_video, save_video
from trackers import (HockeyTracker, VolleyTracker, CricketTracker, BallTracker)
from detectors import ShotDetector
import cv2
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
import numpy as np
import argparse

def main(inp_f_path, sport, model_path):
    # inp_f_path = "input_video/hockey.mp4"
    video_frames = read_video(inp_f_path)

    if sport == "hockey":    
        hock_model = "models/hockey.pt"
        tracker_stubs = "stubs/track_stubs.pkl"
        cam_mov_stubs = "stubs/camera_movement_stub.pkl"

        # Objects Tracking
        tracker = HockeyTracker(hock_model)
        tracks = tracker.get_object_tracks(video_frames,
                                        read_from_stub=True,
                                        stub_path=tracker_stubs)
        
        tracker.add_position_to_tracks(tracks)    
        tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

        # Camera Movement Estimator
        camera_movement_estimator = CameraMovementEstimator(video_frames[0])
        camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                    read_from_stub=True,
                                                                                    stub_path=cam_mov_stubs)
        camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)

        # Team Assigner
        team_assigner = TeamAssigner()
        team_assigner.assign_team_color(video_frames[0], 
                                        tracks['players'][0])
        
        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(video_frames[frame_num],   
                                                    track['bbox'],
                                                    player_id)
                tracks['players'][frame_num][player_id]['team'] = team 
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
        
        # Player Assigner 
        player_assigner = PlayerBallAssigner()
        team_ball_control= []
        for frame_num, player_track in enumerate(tracks['players']):
            if tracks['ball'][frame_num] != {}:
                ball_bbox = tracks['ball'][frame_num][1]['bbox']
                assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

                if assigned_player != -1:
                    tracks['players'][frame_num][assigned_player]['has_ball'] = True
                    team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
                else:
                    team_ball_control.append(team_ball_control[-1])

        # Team Ball Control
        team_ball_control= np.array(team_ball_control)
        output_video_frames = tracker.draw_annotations(video_frames, tracks,team_ball_control)
        output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

        # Save video
        save_video(output_video_frames, 'output_videos/output_video.avi')
    
    elif sport == "volleyball":
        volley_model = "models/volley.pt"
        volley_stubs = "stubs/rack_stubs.pkl"
        cam_mov_stubs = "stubs/camera_movement_stub.pkl"

        # Volley Tracker
        tracker = VolleyTracker(volley_model)

        tracks = tracker.get_object_tracks(video_frames,
                                        read_from_stub=True,
                                        stub_path=volley_stubs)
        

        # for track_id,player in tracks['players'][84].items():

        #     bbox = player['bbox']
        #     frame = video_frames[84]

        #     cropped_image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        #     cv2.imwrite(f'output_videos/cropped_image.jpg',cropped_image)
        #     break
        
        tracker.add_position_to_tracks(tracks)
        
        # tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

        camera_movement_estimator = CameraMovementEstimator(video_frames[0])
        camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                    read_from_stub=True,
                                                                                    stub_path=cam_mov_stubs)
        camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)
        
        view_transformer = ViewTransformer()
        view_transformer.add_transformed_position_to_tracks(tracks)

        speed_and_distance_estimator = SpeedAndDistance_Estimator()
        speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

        team_assigner = TeamAssigner()
        team_assigner.assign_team_color(video_frames[0], 
                                        tracks['players'][0])
        
        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(video_frames[frame_num],   
                                                    track['bbox'],
                                                    player_id)
                tracks['players'][frame_num][player_id]['team'] = team 
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
        

        output_video_frames = tracker.draw_annotations(video_frames, tracks)
        
        speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)

        output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

        save_video(output_video_frames, 'output_videos/output_video.avi')
    
    else:
        crick_model = "models/all.pt"
        balltr_model = "models/balltrack.pt"
        shotsel_model = "models/shotsel.pt"

        player_track = CricketTracker(model_path=crick_model)
        player_detections = player_track.detect_frames(video_frames)

        ball_track = BallTracker(model_path=balltr_model)
        ball_detections = ball_track.detect_frames(video_frames)

        shot_detect = ShotDetector(model_path=shotsel_model)
        shot_detections = shot_detect.detect_frames(video_frames)

        output_video_frames = player_track.draw_bboxes(video_frames, player_detections)
        output_video_frames = ball_track.draw_bboxes(output_video_frames, ball_detections)
        output_video_frames = shot_detect.draw_bboxes(output_video_frames, shot_detections)

        save_video(output_video_frames, "output/tracking.avi")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get task args')

    # Add arguments
    parser.add_argument('--sport', help='Type of Sport', default="hockey")
    parser.add_argument('--inp_f_path', help='Input File Path')
    parser.add_argument('--model_path', help='Model Path')

    args = parser.parse_args()

    main(inp_f_path=args.inp_f_path, sport=args.sport, model_path=args.model_path)