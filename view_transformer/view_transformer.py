import numpy as np
import cv2

class ViewTransformer():
    def __init__(self):
        # court is 68 meters wide
        court_width = 68
        # each division of the field (light green/dark green) is 5.8333meters, 4 divisons = 23.32 meters
        court_length = 23.32
        
        # trapezoid shape, field perspective, selected manually be finding pixel values of points that we want to create the shape with
        self.pixel_verticies = np.array([
            [110,1035],
            [26, 275],
            [910, 260],
            [1640,915]
            ])
        
        # target transformed shape is a rectangle with the given field dimesions, real world verticies
        self.target_verticies = np.array([
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ])
        
        
        self.pixel_verticies = self.pixel_verticies.astype(np.float32)
        self.target_verticies = self.target_verticies.astype(np.float32)
        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_verticies, self.target_verticies)
    
    def transform_point(self, point):
        """ Calculate the position of a point relative to the real world. """
        
        p = (int(point[0]),int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_verticies,p,False) >= 0
        if not is_inside:
            return None
        
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        transform_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
        return transform_point.reshape(-1,2)
        
        
    def add_transformed_position_to_tracks(self, tracks):
        """ Add real world positions the tracks dict based on the perspective transform"""
        
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    position = np.array(position)
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed