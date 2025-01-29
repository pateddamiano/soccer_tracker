from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator
import numpy as np

def main():
    # read video
    video_frames = read_video('input_videos\\08fd33_4.mp4')
    
    # Initialize tracker
    tracker = Tracker('models\\trainrun3_150epoch_best.pt')
    
    # Get object tracks
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs\\track_stubs.pkl')
    
        
    # get object positions for all the tracks
    tracker.add_position_to_tracks(tracks)
    
    # camera movement estimation
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path='stubs\\camera_movement_stub.pkl')
    # adjust player positions per frame based on camera movement
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    
    # View transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    
    # Interpolate missing ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
     
    # assign player teams
    team_assigner =  TeamAssigner()
    # get colors of players in the first frame
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    # assign player to a team based on color
    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track, in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track["bbox"],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
            
    # Assign ball aquisition
    player_assigner = PlayerBallAssigner()
    # create list to count each frame, which team has the ball
    team_ball_control = []
    # for each frame, assign the ball to a player and add to the list containing team ball control stats
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        # if no player has the ball, assign ball to team that last had the ball (in case of pass, etc.)
        else:
            if not team_ball_control:
                team_ball_control.append(0)
            else:
                team_ball_control.append(team_ball_control[-1])
            
    team_ball_control = np.array(team_ball_control)
            
    # draw output 
    ## draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    
    ## Draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    ## Draw speed and distance estimator
    output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)
        
    # save video
    save_video(output_video_frames, 'output_videos\\output.avi')
    
if __name__ == '__main__':
    main()
    
    
    
    # # save cropped image of a player to run kmeans experiment
    # for track_id, player in tracks["players"][0].items():
    #     bbox=player["bbox"]
    #     frame=video_frames[0]
        
    #     # get crop from bounding box
    #     cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
    #     # save cropped bounding box image
    #     cv2.imwrite(f'output_videos\\cropped_image.jpg', cropped_image)
    #     break
    