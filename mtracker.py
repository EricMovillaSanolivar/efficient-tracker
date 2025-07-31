import os
import json
import threading
'''
track_sample = {
    stack: [
        {
            id: Track Id
            area: percentage_area
            position: {x, y}
            area_scale: { scale, time_unit },
            direction_change: { x, y, time_unit }
            timestamp: last updated timestamp
        }
    ]
}
'''

class Mtracker:
    
    def __init__(self, timeout = 5):
        try:
            # Validate directory created
            if not os.path.exists("./tracker"):
                os.makedirs("tracker")
            # Track path
            self._track_path = "./tracker/tracks.json"
            # Load or create new track array
            self._max_id = {}
            self._tracks = self._load_history()
            # Set timeout
            self._timeout = timeout
            print("Tracker initialized...")
        except Exception as err:
            print(err)
            raise ValueError(f"Failed to create tracker due to: {err}")
        
    def setTracks(self, source, tracks):
        if tracks is None:
            return
        # Set tracks
        self._tracks[source] = tracks
    
    def getTracks(self, source):
        """
        Get current tracks
        Returns:
            dict: Dictionary with all tracks
        """
        return self._tracks[source]
    
    def _iou(self, bx1, bx2):
        # Get max and mins
        il = max(bx1[0], bx2[0])
        ir = min(bx1[2], bx2[2])
        it = max(bx1[1], bx2[1])
        ib = min(bx1[3], bx2[3])
        # Return 0 if no intersection otherwise calculate intersection
        return 0 if ir < il or ib < it else (ir - il) * (ib - it)
    
    
    def _load_history(self):
        # Verify if the file exists
        if not os.path.exists(self._track_path):
            # Create an empty dictionary
            with open(self._track_path, "w") as file:
                json.dump({}, file)
            return {}

        # Attempt to read existent file
        with open(self._track_path, "r") as file:
            try:
                res = json.load(file)
                # Attempt to update last track id used
                for src in res.keys():
                    self._max_id[src] = 0
                    if len(res[src]) > 0:
                        for trk in res[src]:
                            if "track_id" in trk and self._max_id[src] < trk["track_id"]:
                                self._max_id[src] = trk["track_id"]
                        self._max_id[src] += 1
                return res
            except json.JSONDecodeError:
                # Manage non valid json
                with open(self._track_path, "w") as fw:
                    json.dump({}, fw)
                return {}
            
    def _update_history(self):
        try:
            with open(self._track_path, "w") as file:
                json.dump(self._tracks, file, indent=4)
        except Exception as err:
            raise ValueError("Mtracker: Failed to update track history")
        
    def update(self, source, detections, time_stamp):
        try:
            # Validate current tracks
            if not source in self._tracks:
                print(f"Creating source {source}")
                self._tracks[source] = []
            # Validate current tracks
            if not source in self._max_id:
                self._max_id[source] = 0
            if len(self._tracks[source]) == 0: # There are not current tracks
                # Add every detection as a new track
                for index, det in enumerate(detections):
                    det["id"] = index # Embed track id to results
                    # Reference to detection bbox
                    bbx = det["bbox"]
                    # Add track
                    self._tracks[source].append({
                        "track_id": self._max_id[source],
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
                    self._max_id[source] += 1
                # Return detections with assignded track_ids
                return detections
            else:
                # Results reference
                results = []
                # Delta time
                tracks_to_remove = []
                # Search for class matches
                for trk_ind, track in enumerate(self._tracks[source]):
                    delta_time = time_stamp - track["timestamp"]
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
                        self._tracks[source][trk_ind] = track_match.copy()
                    else:
                        # Kill ids
                        if delta_time > self._timeout:
                            print(f"track {track['track_id']} deleted")
                            tracks_to_remove.append(trk_ind)
                
                # Delete expired indexes
                for idx in reversed(tracks_to_remove):
                    self._tracks[source].pop(idx)
                
                # Add ids to remaining detections
                if len(detections) > 0:
                    for det in detections:
                        # Assign track id
                        det["id"] = self._max_id[source]
                        # Add to results
                        results.append(det)
                        # Add track
                        # Reference to detection bbox
                        bbx = det["bbox"]
                        self._tracks[source].append({
                            "track_id": self._max_id[source],
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
                        self._max_id[source] += 1
                return results
        except Exception as err:
            raise ValueError(f"Mtracker: Error while trying to update tracks: {err}")
        finally:
            threading.Thread(target=self._update_history, daemon=False).start()