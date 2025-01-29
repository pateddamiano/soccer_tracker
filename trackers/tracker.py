from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    
    def add_position_to_tracks(self, tracks):
        """ add positions of all object tracks to the tracks """
        # loop through all items, ball, players, and referees
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info["bbox"]
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position
    
    def interpolate_ball_positions(self, ball_positions):
        """ Interpolate the position of the ball in frames where the ball is not detected"""
        # convert the ball bbox from every detected frame into a list, add an entry with an empty list for frames where the ball is not detected
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1', 'y1', 'x2', 'y2'])
        
        #interpolate missing ball bboxes
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        
        # put ball positions from data frame back to original format
        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]
        
        return ball_positions
        
    def detect_frames(self, frames):
        """ Run YOLO on the list of frames. """
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """ Run detect_frames() to get detections and run through supervision object tracker to get tracks. """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            print(f'Tracks loaded from {stub_path}')
            return tracks
            
        # Get Detections
        detections = self.detect_frames(frames)
        
        # create dict to hold all tracks of objects
        tracks = {
            "players":[],
            # frame 0: { 0:{"bbox":[x1, y1, x2, y2]}, 1:{bbox:[x1, y1, x2, y2]}}
            # frame 1: { 0:{"bbox":[x1, y1, x2, y2]}, 1:{bbox:[x1, y1, x2, y2]}}
            # etc. etc...
            "referees":[],
            "ball":[]
        }
        
        # overwrite goalkeeper as a player
        for frame_num, detection in enumerate(detections):

            # 0:ball, 1:goalkeeper, 2:player, 3:referee
            cls_names = detection.names
            # ball:0, goalkeeper:1, player:2, referee:3
            cls_names_inv = {v:k for k,v in cls_names.items()}
            
            # convert detections to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            # Convert goalkeeper to player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    # swap each class id of goalkeeper '1' to player id '2'
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]
            
            # track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            # add tracked objects to tracks dict
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            # get player and ref tracks
            for frame_detection in detection_with_tracks:
                # boxes are 1st index in the supervision detection tracker format
                bbox = frame_detection[0].tolist()
                # class ID list is 4th index in the supervision detection tracker format
                cls_id = frame_detection[3]
                # track ID list is 5th index in the supervision detection tracker format
                track_id = frame_detection[4]
                
                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            # since there is only one ball, get tracks directly from the supervision detections
            for frame_detection in detection_supervision:    
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                
                if class_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}
            
            # save tracks to a file for quick loading
            if stub_path is not None:
                with open(stub_path, 'wb') as f:
                    pickle.dump(tracks,f)

        return tracks
        
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        """ Draw ellipse to signify player at bottom of bbox. """
        # y2 is last coordinate in the bbox
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        
        cv2.ellipse(frame, 
            center=(x_center, y2), 
            axes=(int(width), int(0.35*width)),
            angle=0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4 
        ) 
        
        rectangle_width = 40
        rectange_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectange_height // 2) + 15
        y2_rect = (y2 + rectange_height // 2) + 15
        
        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -=10
                
            cv2.putText(
                frame,
                f"{track_id}", # text
                (int(x1_text), int(y1_rect+15)), # location
                cv2.FONT_HERSHEY_SIMPLEX, # font
                0.6, # size
                (0, 0, 0), # color, black
                thickness = 2
            )
        
        return frame
        
    def draw_triangle(self, frame, bbox, color):
        """ Draw triangle to point to ball """
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        triangle_points = np.array([
            [x,y],
            [x-10, y-20],
            [x+10, y-20]
        ])
        
        cv2.drawContours(
            frame,
            [triangle_points],
            0,
            color,
            cv2.FILLED
        )
        cv2.drawContours(
            frame,
            [triangle_points],
            0,
            (0, 0, 0), # color
            2 # not filled
        )
        
        return frame
    
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        """ Draw a rectangle in the corner to track team on-ball percentages """
        
        # draw semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255,255,255), cv2.FILLED)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # get ball control for each team in past frames
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        if (team_1_num_frames != 0) or (team_2_num_frames != 0):
            team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames) 
            team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)
        else:
            team_1 = 0
            team_2 = 0
        
        # put text on frames
        cv2.putText(frame, f"Team 1 Ball Control: {team_1 * 100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2 * 100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        
        return frame
          
    def draw_annotations(self, video_frames, tracks, team_ball_control):
        """ Draw annotations of tracks and statistics on video frames. """
        
        output_video_frames=[]
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            player_dict = tracks['players'][frame_num]
            referee_dict = tracks['referees'][frame_num]
            ball_dict = tracks['ball'][frame_num]
            
            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
                
                if player.get('has_ball', False):
                    frame=self.draw_triangle(frame, player['bbox'], (0, 0, 255))
            
            # Draw referees
            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))
            
            # draw ball triangle pointer
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))
                
            # Draw team ball control 
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            
            output_video_frames.append(frame)
            
        return output_video_frames