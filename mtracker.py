import os
import time
import json
'''
track_sample = {
    id: Track Id
    area: percentage_area
    position: {x, y}
    area_scale: { scale, time_unit },
    direction_change: { x, y, time_unit }
    timestamp: last updated timestamp
}
'''

class Mtracker:
    
    def __init__(self, id, tracks = None, timeout = 5000):
        # Validate directory created
        if not os.path.exists("./tracker"):
            os.makedirs("tracker")
        # Set ID
        self._id = id
        # Set timeout
        self._timeout = timeout
        # Load or create new track array
        self._tracks = [] if tracks is None else tracks
        # Set last track update timestamp
        self._last_update = 0
        # Validate existing timestamp
        if tracks is not None:
            for trk in tracks:
                self._last_update = max(self._last_update, trk['timestamp'])
        else:
            self._last_update = time.time()
        
    def setTracks(self, tracks):
        if tracks is None:
            return
        # Set tracks
        self._tracks = tracks
    
    def getTracks(self):
        """
        Get current tracks
        Returns:
            dict: Dictionary with all tracks
        """
        return self._tracks
    
    def _iou(self, bx1, bx2):
        # Get max and mins
        il = max(bx1[0], bx2[0])
        ir = min(bx1[2], bx2[2])
        it = max(bx1[1], bx2[1])
        ib = min(bx1[3], bx2[3])
        # Return 0 if no intersection otherwise calculate intersection
        return 0 if ir < il or ib < it else (ir - il) * (ib - it)
        
    def update(self, detections, time_stamp):
        
        try:
            # Validate current tracks
            if len(self._tracks) == 0: # There are not current tracks
                # Add every detection as a new track
                for index, det in enumerate(detections):
                    det["id"] = index # Embed track id to results
                    # Reference to detection bbox
                    bbx = det["bbox"]
                    # Add track
                    self._tracks.append({
                        "track_id": index,
                        "class_id": det["class_id"],
                        "bbox": bbx,
                        "area": (bbx[2] - bbx[0]) * (bbx[3] - bbx[1]),
                        "area_scale": 1,
                        "position": {
                            "x": bbx[0] + ((bbx[2] - bbx[0]) / 2),
                            "y": bbx[1] + ((bbx[3] - bbx[1]) / 2)
                        },
                        "position_change": {
                            "x": 0,
                            "y": 0
                        },
                        "delta_time": 1,
                        "timestamp": time_stamp
                    })
                # Return detections with assignded track_ids
                return detections
            else:
                # Results reference
                results = []
                # matches = []
                # Delta time
                delta_time = time_stamp - self._last_update
                # Search for class matches
                for trk_ind, track in enumerate(self._tracks):
                    pred_area = (delta_time * track["area_scale"] / track["delta_time"]) * track["area"]
                    pred_pos = {
                        "x": ((delta_time * track["position_change"]["x"]) / track["delta_time"]) + track["position"]["x"],
                        "y": ((delta_time * track["position_change"]["y"]) / track["delta_time"]) + track["position"]["y"]
                    }
                    track_match = {}
                    # Loop thru all detections
                    min_match_score = 1000
                    min_match_index = 0
                    match_found = False
                    for index, det in enumerate(detections):
                        if det["class_id"] == track["class_id"]:
                            match_found = True
                            bx = det["bbox"]
                            # detection area
                            det_area = (bx[2] - bx[0]) * (bx[3] - bx[1])
                            # detection position
                            det_pos = {
                                "x": bx[0] + ((bx[2] - bx[0]) / 2),
                                "y": bx[1] + ((bx[3] - bx[1]) / 2)
                            }
                            
                            match_score = (abs(pred_pos["x"] - det_pos["x"]) + abs(pred_pos["y"] - det_pos["y"]) + abs(pred_area - det_area)) * (1 - self._iou(track["bbox"], det["bbox"]))
                            
                            if match_score < min_match_score:
                                min_match_index = index
                                min_match_score = match_score
                                track_match = {
                                    "class_id": det["class_id"],
                                    "track_id": track["track_id"],
                                    "bbox": bx,
                                    "area": det_area,
                                    "area_scale": det_area / track["area"],
                                    "position": det_pos,
                                    "position_change": {
                                        "x": det_pos["x"] - track["position"]["x"],
                                        "y": det_pos["y"] - track["position"]["y"]
                                    },
                                    "delta_time": delta_time,
                                    "timestamp": time_stamp
                                }
                    if match_found:
                        # Detection result
                        result = detections[min_match_index].copy()
                        # Delete from input detections
                        del detections[min_match_index]
                        # Add track id
                        result["id"] = int(track["track_id"])
                        # Add result with assigned track_id
                        results.append(result)
                        # Update track
                        self._tracks[trk_ind] = track_match.copy()
                    else:
                        # Kill ids
                        if time_stamp - track["timestamp"] > self._timeout:
                            print(f"track {trk_ind} deleted")
                            self._tracks.pop(trk_ind)
                            
                
                # Add ids to remaining detections
                if len(detections) > 0:
                    trks = self._tracks
                    start_id = trks[len(trks) - 1]["track_id"] + 1
                    for det in detections:
                        # Assign track id
                        det["id"] = start_id
                        # Add to results
                        results.append(det)
                        # Add track
                        # Reference to detection bbox
                        bbx = det["bbox"]
                        self._tracks.append({
                            "track_id": start_id,
                            "class_id": det["class_id"],
                            "bbox": bbx,
                            "area": (bbx[2] - bbx[0]) * (bbx[3] - bbx[1]),
                            "area_scale": 1,
                            "position": {
                                "x": bbx[0] + ((bbx[2] - bbx[0]) / 2),
                                "y": bbx[1] + ((bbx[3] - bbx[1]) / 2)
                            },
                            "position_change": {
                                "x": 0,
                                "y": 0
                            },
                            "delta_time": 1,
                            "timestamp": time_stamp
                        })
                        # Increase id count
                        start_id += 1
                return results
        except Exception as err:
            print(f"Error while trying to update tracks: {err}")
            return detections
        finally:
            with open(f"./tracker/tracks-{self._id}.json", "w") as file:
                file.write(json.dumps({ "tracks": self._tracks }))