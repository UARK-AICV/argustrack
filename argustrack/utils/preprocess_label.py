from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment
from .multicamera_matching import MultiCameraTracking
from .object_tracking import ObjectTracking


class FixedIDTracker:
    """
    Offline tracker with fixed number of objects.
    Uses Hungarian assignment on BEV footpoints (representative points from clusters) frame-to-frame to minimize ID switches.
    Note: Uses actual detection points (footpoints) from clusters, not computed centroids.
    """

    def __init__(self, list_of_labels: List[List[Tuple[float, float]]], expected_objects: int, max_distance: float):
        self.list_of_labels = list_of_labels
        self.expected_objects = expected_objects
        self.max_distance = max_distance
        self.track_states = {}  # track_id -> {'pos': (x,y) or None}

    def _init_tracks(self, detections: List[Tuple[float, float]]):
        # Assign IDs 0..expected_objects-1 based on first frame detections
        mapping = {}
        num_init = min(len(detections), self.expected_objects)
        for det_idx in range(num_init):
            self.track_states[det_idx] = {'pos': detections[det_idx]}
            mapping[det_idx] = det_idx
        # If fewer detections than expected, create empty tracks
        for tid in range(num_init, self.expected_objects):
            self.track_states[tid] = {'pos': None}
        return mapping

    def _build_cost(self, detections: List[Tuple[float, float]]):
        # Cost matrix shape: tracks x detections
        if not detections:
            return None
        cost = np.full((self.expected_objects, len(detections)), fill_value=self.max_distance * 10, dtype=np.float32)
        for tid in range(self.expected_objects):
            tpos = self.track_states[tid]['pos']
            if tpos is None:
                continue
            for did, dpos in enumerate(detections):
                dx = tpos[0] - dpos[0]
                dy = tpos[1] - dpos[1]
                dist = np.sqrt(dx * dx + dy * dy)
                cost[tid, did] = dist
        return cost

    def assign_ids(self) -> Dict[Tuple[int, int], int]:
        """
        Returns mapping {(frame_idx, obj_idx): global_id}
        obj_idx corresponds to index in self.list_of_labels[frame_idx]
        """
        if self.expected_objects is None or self.expected_objects <= 0:
            return {}

        temporal_map: Dict[Tuple[int, int], int] = {}

        if not self.list_of_labels:
            return temporal_map

        # Initialize with first frame
        first_dets = self.list_of_labels[0] if len(self.list_of_labels) > 0 else []
        init_map = self._init_tracks(first_dets)
        for det_idx, gid in init_map.items():
            temporal_map[(0, det_idx)] = gid

        # Process subsequent frames
        for f_idx in range(1, len(self.list_of_labels)):
            dets = self.list_of_labels[f_idx]
            if not dets:
                # No detections, keep track states
                continue

            cost = self._build_cost(dets)
            if cost is None:
                continue

            row_ind, col_ind = linear_sum_assignment(cost)

            # Assign matches within max_distance
            assigned_dets = set()
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] <= self.max_distance:
                    self.track_states[r]['pos'] = dets[c]
                    temporal_map[(f_idx, c)] = r
                    assigned_dets.add(c)

            # Unassigned tracks keep previous position
            # Unassigned detections: assign to nearest available track
            # First, try to find tracks with None pos that haven't been assigned in this frame
            for det_idx, det in enumerate(dets):
                if det_idx in assigned_dets:
                    continue
                
                # Find the nearest unassigned track (even if it has a position from previous frame)
                # This ensures all detections are assigned if possible
                best_tid = None
                min_dist = float('inf')
                
                # First, try tracks with None pos
                for tid in range(self.expected_objects):
                    if tid not in [temporal_map.get((f_idx, c), -1) for c in assigned_dets]:
                        tpos = self.track_states[tid]['pos']
                        if tpos is None:
                            # Found a free track
                            best_tid = tid
                            min_dist = 0
                            break
                
                # If no free track, find the nearest track that hasn't been assigned in this frame
                if best_tid is None:
                    for tid in range(self.expected_objects):
                        # Check if this track was already assigned in this frame
                        already_assigned = any(temporal_map.get((f_idx, c), -1) == tid for c in assigned_dets)
                        if not already_assigned:
                            tpos = self.track_states[tid]['pos']
                            if tpos is not None:
                                dx = tpos[0] - det[0]
                                dy = tpos[1] - det[1]
                                dist = np.sqrt(dx * dx + dy * dy)
                                if dist < min_dist:
                                    min_dist = dist
                                    best_tid = tid
                
                # Assign to best track if found and within max_distance
                if best_tid is not None and min_dist <= self.max_distance * 2:  # Allow larger distance for unassigned detections
                    self.track_states[best_tid]['pos'] = det
                    temporal_map[(f_idx, det_idx)] = best_tid
                # else drop this detection (should rarely happen)

        return temporal_map


class PreprocessLabel:
    """
    Combine multiview matching and temporal tracking to assign global IDs
    to objects across multiple cameras and frames.
    
    Process:
    1. For each frame: match objects across cameras (multiview matching)
    2. Track matched objects across frames (temporal tracking)
    3. Output: final global_id for each (frame_idx, camera_name, local_id)
    """
    
    def __init__(self, calib_folder: str, max_distance_multiview: float = 2.0, 
                 max_distance_tracking: float = 50.0, n_clusters: int = None,
                 max_centroid_movement: float = 5.0, smoothing_factor: float = 0.3,
                 intrinsic_scale: float = 0.25, translation_scale: float = 100.0,
                 bev_x: int = None, bev_y: int = None, bev_bounds: list = None):
        """
        Args:
            calib_folder: Path to calibration folder containing intrinsic/extrinsic
            max_distance_multiview: Maximum distance in BEV space for multiview clustering (meters) - deprecated, not used with KMeans
            max_distance_tracking: Maximum distance in BEV space for temporal tracking (meters)
            n_clusters: Number of clusters for KMeans = fixed number of objects in the scene.
                       For offline tracking, this should be set to the exact number of objects.
                       KMeans will create exactly n_clusters clusters (some may be empty if fewer detections).
                       If None, will be auto-determined from number of detections (not recommended for offline tracking)
            max_centroid_movement: Maximum allowed movement of cluster centers between frames (meters) for stability
            smoothing_factor: Smoothing factor for cluster centers (0.0 = no smoothing, 1.0 = full smoothing)
                             Lower values mean more responsive to changes, higher values mean more stable
            intrinsic_scale: Scale factor for intrinsic matrix (default 0.25 = 1/4)
            translation_scale: Scale factor for translation vector (default 100.0)
            bev_x: BEV grid X dimension (default from constants)
            bev_y: BEV grid Y dimension (default from constants)
            bev_bounds: BEV bounds [XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX] (default from constants)
        """
        from argustrack.utils.constants import DEFAULT_INTRINSIC_SCALE, DEFAULT_TRANSLATION_SCALE
        from argustrack.utils.constants import DEFAULT_BEV_X, DEFAULT_BEV_Y, DEFAULT_BEV_BOUNDS
        
        self.calib_folder = calib_folder
        self.max_distance_multiview = max_distance_multiview
        self.max_distance_tracking = max_distance_tracking
        self.n_clusters = n_clusters
        self.max_centroid_movement = max_centroid_movement
        self.smoothing_factor = smoothing_factor
        
        # Use provided scale factors or defaults from constants
        self.intrinsic_scale = intrinsic_scale if intrinsic_scale is not None else DEFAULT_INTRINSIC_SCALE
        self.translation_scale = translation_scale if translation_scale is not None else DEFAULT_TRANSLATION_SCALE
        self.bev_x = bev_x if bev_x is not None else DEFAULT_BEV_X
        self.bev_y = bev_y if bev_y is not None else DEFAULT_BEV_Y
        self.bev_bounds = bev_bounds if bev_bounds is not None else DEFAULT_BEV_BOUNDS.copy()
        
        # Initialize multiview matcher with scale factors
        self.multiview_matcher = MultiCameraTracking(
            calib_folder=calib_folder,
            max_distance=max_distance_multiview,
            n_clusters=n_clusters,
            intrinsic_scale=self.intrinsic_scale,
            translation_scale=self.translation_scale,
            bev_x=self.bev_x,
            bev_y=self.bev_y,
            bev_bounds=self.bev_bounds
        )
        
        # Final mapping: (frame_idx, camera_name, local_id) -> final_global_id
        self.final_global_id_map = {}
        
        # Store frame BEV coordinates (centroids) for later access
        self.frame_bev_coords = {}  # {frame_idx: {frame_global_id: (x, y)}}
        
    def preprocess(self, multi_camera_detections: Dict[str, List[List[List[float]]]]) -> Dict[Tuple[int, str, int], int]:
        """
        Process multi-camera detections across frames to assign final global IDs
        
        Args:
            multi_camera_detections: Dict mapping camera_name -> list of frames,
                                   each frame is list of bboxes [x_min, y_min, x_max, y_max]
            
        Returns:
            Dict mapping (frame_idx, camera_name, local_id) -> final_global_id
        """
        if not multi_camera_detections:
            return {}
        
        # Get number of frames from first camera
        camera_names = list(multi_camera_detections.keys())
        if not camera_names:
            return {}
        
        num_frames = len(multi_camera_detections[camera_names[0]])
        
        # Verify all cameras have the same number of frames
        # If any camera has fewer frames, pad with empty lists
        # If any camera has more frames, trim to num_frames
        for cam_name in camera_names:
            cam_frames = len(multi_camera_detections[cam_name])
            if cam_frames < num_frames:
                print(f"WARNING: Camera {cam_name} has {cam_frames} frames, padding to {num_frames}")
                # Pad with empty detection lists
                while len(multi_camera_detections[cam_name]) < num_frames:
                    multi_camera_detections[cam_name].append([])
            elif cam_frames > num_frames:
                print(f"WARNING: Camera {cam_name} has {cam_frames} frames, trimming to {num_frames}")
                # Trim to num_frames
                multi_camera_detections[cam_name] = multi_camera_detections[cam_name][:num_frames]
            elif cam_frames != num_frames:
                print(f"WARNING: Camera {cam_name} has {cam_frames} frames, expected {num_frames}")
        
        print(f"Processing {len(camera_names)} cameras: {camera_names}")
        print(f"Total frames: {num_frames} (will process frames 0 to {num_frames-1})")
        
        # Step 1: Multiview matching for each frame
        # frame_multiview_map: {frame_idx: {(camera_name, local_id): frame_global_id}}
        frame_multiview_maps = {}
        # frame_bev_coords: {frame_idx: {frame_global_id: (x, y)}} - centroid of cluster
        frame_bev_coords = {}
        
        print(f"Step 1: Multiview matching for {num_frames} frames...")
        print(f"  Will process frames: {list(range(num_frames))}")
        prev_smoothed_centroids = None  # Track smoothed centroids for next frame
        
        for frame_idx in range(num_frames):
            if frame_idx == num_frames - 1:
                print(f"  Processing last frame: {frame_idx}")
            # Get detections for this frame - ensure ALL cameras are included
            frame_detections = {}
            total_detections_in_frame = 0
            for camera_name in camera_names:
                if frame_idx < len(multi_camera_detections[camera_name]):
                    frame_detections[camera_name] = multi_camera_detections[camera_name][frame_idx]
                    total_detections_in_frame += len(frame_detections[camera_name])
                else:
                    frame_detections[camera_name] = []
            
            # Verify all cameras are in frame_detections
            if len(frame_detections) != len(camera_names):
                missing_cams = set(camera_names) - set(frame_detections.keys())
                print(f"WARNING: Frame {frame_idx} missing cameras: {missing_cams}")
                # Add missing cameras with empty detections
                for cam_name in missing_cams:
                    frame_detections[cam_name] = []
            
            # Debug: Log detections per camera for first frame
            if frame_idx == 0:
                print(f"  Frame {frame_idx} detections per camera:")
                for cam_name in camera_names:
                    num_dets = len(frame_detections.get(cam_name, []))
                    print(f"    {cam_name}: {num_dets} detections")
                print(f"  Total detections in frame: {total_detections_in_frame}")
            
            # Match multiview for this frame with warm start from previous frame
            multiview_map = self.multiview_matcher.match_multiview(
                frame_detections,
                prev_centroids=prev_smoothed_centroids,
                max_centroid_movement=self.max_centroid_movement
            )
            frame_multiview_maps[frame_idx] = multiview_map
            
            # Debug: Log multiview matching results for first frame
            if frame_idx == 0:
                print(f"  Frame {frame_idx} multiview matching results:")
                # Count detections per global_id
                global_id_counts = {}
                for (cam_name, local_id), global_id in multiview_map.items():
                    if global_id not in global_id_counts:
                        global_id_counts[global_id] = []
                    global_id_counts[global_id].append((cam_name, local_id))
                print(f"    Found {len(global_id_counts)} clusters:")
                for global_id in sorted(global_id_counts.keys()):
                    members = global_id_counts[global_id]
                    print(f"      Cluster {global_id}: {len(members)} detections from cameras {[m[0] for m in members]}")
            
            # Collect BEV coordinates for each frame_global_id (representative footpoint from cluster)
            frame_global_id_coords = {}  # {frame_global_id: (x, y)}
            
            # IMPORTANT: Use representative footpoint from multiview_matcher instead of computing centroid
            # The footpoint is selected as the point closest to the cluster centroid (one actual detection point)
            current_frame_centroids = {}  # {frame_global_id: np.array([x, y])} - stores footpoint, not centroid
            
            # Get representative footpoints from multiview_matcher (computed in _cluster_multiview_detections)
            for frame_global_id in set(multiview_map.values()):
                footpoint = self.multiview_matcher.get_cluster_centroid(frame_global_id)
                if footpoint is not None:
                    x, y = footpoint
                    if np.isfinite(x) and np.isfinite(y):
                        current_frame_centroids[frame_global_id] = np.array([x, y])
                        # Round footpoint to ensure consistency (~1 cm precision)
                        footpoint_rounded = (round(float(x), 2), round(float(y), 2))
                        frame_global_id_coords[frame_global_id] = footpoint_rounded
                        
                        # Debug: Print footpoint for first frame
                        if frame_idx == 0:
                            # Count points in this cluster for debug info
                            cluster_points_count = sum(1 for fgid in multiview_map.values() if fgid == frame_global_id)
                            print(f"  Frame {frame_idx}, cluster {frame_global_id}: footpoint = ({footpoint_rounded[0]:.2f}, {footpoint_rounded[1]:.2f}) from {cluster_points_count} detections")
                else:
                    # Fallback: select first point from cluster if footpoint not available
                    # This should rarely happen, but provides a safety net
                    frame_global_id_points = []
                    for (camera_name, local_id), fgid in multiview_map.items():
                        if fgid == frame_global_id:
                            bev_coord = self.multiview_matcher.get_bev_coords(camera_name, local_id)
                            if bev_coord is not None:
                                x, y = bev_coord
                                if np.isfinite(x) and np.isfinite(y):
                                    frame_global_id_points.append((x, y))
                    
                    if frame_global_id_points:
                        # Use first point as representative footpoint
                        footpoint = frame_global_id_points[0]
                        current_frame_centroids[frame_global_id] = np.array(footpoint)
                        footpoint_rounded = (round(float(footpoint[0]), 2), round(float(footpoint[1]), 2))
                        frame_global_id_coords[frame_global_id] = footpoint_rounded
                        if frame_idx == 0:
                            print(f"  Frame {frame_idx}, cluster {frame_global_id}: fallback footpoint = ({footpoint_rounded[0]:.2f}, {footpoint_rounded[1]:.2f}) from {len(frame_global_id_points)} points")
            
            # Apply smoothing with previous centroids for better stability
            # Note: smoothing is only applied after merge and remap IDs to ensure consistency
            # Temporarily store current centroids to use after merge
            temp_current_centroids = current_frame_centroids.copy()
            
            # For offline tracking: ensure we have exactly n_clusters clusters.
            # With KMeans, n_clusters is fixed and KMeans always creates exactly n_clusters clusters.
            # However, after computing and rounding centroids, there may be fewer clusters due to duplicates.
            # In rare cases there may appear to be more clusters; we guard against that here.
            # If there are more than n_clusters clusters, merge the closest ones.
            # If there are fewer than n_clusters clusters, make sure cluster IDs are still numbered from 0 to n_clusters-1.
            if self.n_clusters is not None:
                num_current_clusters = len(frame_global_id_coords)
                
                if frame_idx == 0:
                    print(f"  Frame {frame_idx}: n_clusters={self.n_clusters}, num_current_clusters={num_current_clusters}")
                
                # KMeans always creates exactly n_clusters clusters, but after centroid rounding
                # we may end up with fewer due to duplicates (or more in edge cases).
                if num_current_clusters > self.n_clusters:
                    if frame_idx == 0:
                        print(f"  Frame {frame_idx}: Found {num_current_clusters} clusters, merging to {self.n_clusters}...")
                    
                    # Merge closest clusters until we have exactly n_clusters
                    merge_count = 0
                    while len(frame_global_id_coords) > self.n_clusters:
                        # Find the two closest centroids
                        coords_list = list(frame_global_id_coords.items())
                        min_dist = float('inf')
                        merge_pair = None
                        
                        for i in range(len(coords_list)):
                            for j in range(i + 1, len(coords_list)):
                                id1, coord1 = coords_list[i]
                                id2, coord2 = coords_list[j]
                                dist = np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
                                if dist < min_dist:
                                    min_dist = dist
                                    merge_pair = (id1, id2)
                        
                        # Merge the closest pair
                        if merge_pair:
                            id1, id2 = merge_pair

                        # Before merging, check constraint: the merged cluster
                        # must not contain two detections from the same camera.
                            cams_in_id1 = set()
                            cams_in_id2 = set()
                            for (cam_name, local_id), frame_global_id in multiview_map.items():
                                if frame_global_id == id1:
                                    cams_in_id1.add(cam_name)
                                elif frame_global_id == id2:
                                    cams_in_id2.add(cam_name)

                            # If the two clusters share any camera, merging would violate the constraint -> skip
                            if cams_in_id1 & cams_in_id2:
                                # No valid pairs remain that satisfy the camera-uniqueness constraint
                                break

                            # Merge all detections from id2 into id1
                            for (cam_name, local_id), frame_global_id in list(multiview_map.items()):
                                if frame_global_id == id2:
                                    multiview_map[(cam_name, local_id)] = id1
                            
                            # Update centroids: remove id2, keep id1
                            # Recompute centroid for id1 with all points from both clusters
                            merged_points = []
                            for (cam_name, local_id), fgid in multiview_map.items():
                                if fgid == id1:
                                    bev_coord = self.multiview_matcher.get_bev_coords(cam_name, local_id)
                                    if bev_coord is not None:
                                        x, y = bev_coord
                                        if np.isfinite(x) and np.isfinite(y):
                                            merged_points.append((x, y))
                            
                            if merged_points:
                                points_array = np.array(merged_points)
                                centroid = np.mean(points_array, axis=0)
                                # Round centroid again to ensure consistency
                                centroid_rounded = (round(float(centroid[0]), 2), round(float(centroid[1]), 2))
                                frame_global_id_coords[id1] = centroid_rounded
                            
                            del frame_global_id_coords[id2]
                            frame_multiview_maps[frame_idx] = multiview_map
                            merge_count += 1
                        else:
                            break  # No more pairs to merge
                    
                    if frame_idx == 0 and merge_count > 0:
                        print(f"  Merged {merge_count} clusters in frame {frame_idx}")
                
                # Ensure cluster IDs are numbered from 0 to n_clusters-1.
                # Remap frame_global_id to keep IDs contiguous from 0 up to n_clusters-1.
                # If there are fewer clusters than n_clusters, some IDs will be unused,
                # which is acceptable for temporal tracking.
                if len(frame_global_id_coords) > 0:
                    # Create mapping from old frame_global_id to new (0 .. num_actual_clusters-1)
                    old_ids = sorted(frame_global_id_coords.keys())
                    num_actual_clusters = len(old_ids)
                    # Remap so that IDs range from 0 to num_actual_clusters-1.
                    # If num_actual_clusters < n_clusters, some IDs will remain unused.
                    id_remap = {old_id: new_id for new_id, old_id in enumerate(old_ids)}
                    
                    # Remap trong multiview_map
                    new_multiview_map = {}
                    for (cam_name, local_id), old_frame_global_id in multiview_map.items():
                        new_frame_global_id = id_remap.get(old_frame_global_id, old_frame_global_id)
                        new_multiview_map[(cam_name, local_id)] = new_frame_global_id
                    multiview_map = new_multiview_map
                    
                    # Remap trong frame_global_id_coords
                    new_frame_global_id_coords = {}
                    for old_id, coord in frame_global_id_coords.items():
                        new_id = id_remap.get(old_id, old_id)
                        new_frame_global_id_coords[new_id] = coord
                    frame_global_id_coords = new_frame_global_id_coords
                    
                    frame_multiview_maps[frame_idx] = multiview_map
                    
                    # Log a warning if we ended up with fewer clusters than n_clusters
                    if num_actual_clusters < self.n_clusters and frame_idx == 0:
                        print(f"  Frame {frame_idx}: Found {num_actual_clusters} clusters (expected {self.n_clusters})")
            
            # Apply smoothing with previous centroids after merge and remap
            if prev_smoothed_centroids is not None and self.n_clusters is not None and len(frame_global_id_coords) > 0:
                # Create array of current centroids sorted by ID
                sorted_ids = sorted(frame_global_id_coords.keys())
                current_centroids_array = np.array([frame_global_id_coords[fid] for fid in sorted_ids])
                
                # Ensure prev_smoothed_centroids has the same count as current
                if len(prev_smoothed_centroids) == len(current_centroids_array):
                    # Match current with previous using Hungarian algorithm
                    distances = np.linalg.norm(
                        current_centroids_array[:, None, :] - prev_smoothed_centroids[None, :, :],
                        axis=2
                    )
                    row_ind, col_ind = linear_sum_assignment(distances)
                    
                    # Apply smoothing: smoothed = (1-alpha) * current + alpha * previous
                    smoothed_centroids_array = np.zeros_like(current_centroids_array)
                    # row_ind and col_ind have the same length, each element i: row_ind[i] -> col_ind[i]
                    for match_idx in range(len(row_ind)):
                        current_idx = row_ind[match_idx]
                        prev_idx = col_ind[match_idx]
                        if current_idx < len(current_centroids_array) and prev_idx < len(prev_smoothed_centroids):
                            current_cent = current_centroids_array[current_idx]
                            prev_cent = prev_smoothed_centroids[prev_idx]
                            smoothed_cent = (1 - self.smoothing_factor) * current_cent + self.smoothing_factor * prev_cent
                            smoothed_centroids_array[current_idx] = smoothed_cent
                            
                            # Update frame_global_id_coords with smoothed centroid
                            fid = sorted_ids[current_idx]
                            centroid_rounded = (round(float(smoothed_cent[0]), 2), round(float(smoothed_cent[1]), 2))
                            frame_global_id_coords[fid] = centroid_rounded
                    
                    # Prepare smoothed centroids for next frame
                    # Sort by ID to ensure consistent order
                    prev_smoothed_centroids = smoothed_centroids_array.copy()
                elif len(prev_smoothed_centroids) == self.n_clusters and len(current_centroids_array) <= self.n_clusters:
                    # If previous has n_clusters but current has fewer, only smooth matched ones
                    # Pad current with zeros to match n_clusters
                    padded_current = np.zeros((self.n_clusters, 2))
                    padded_current[:len(current_centroids_array)] = current_centroids_array
                    # Assign large value to positions without current centroid to avoid matching
                    padded_current[len(current_centroids_array):] = np.inf
                    
                    distances = np.linalg.norm(
                        padded_current[:, None, :] - prev_smoothed_centroids[None, :, :],
                        axis=2
                    )
                    # Replace inf with large value for Hungarian algorithm to work
                    distances = np.where(np.isfinite(distances), distances, 1e10)
                    row_ind, col_ind = linear_sum_assignment(distances)
                    
                    smoothed_centroids_array = prev_smoothed_centroids.copy()
                    for match_idx in range(len(row_ind)):
                        current_idx = row_ind[match_idx]
                        prev_idx = col_ind[match_idx]
                        if current_idx < len(current_centroids_array) and prev_idx < len(prev_smoothed_centroids):
                            current_cent = current_centroids_array[current_idx]
                            prev_cent = prev_smoothed_centroids[prev_idx]
                            smoothed_cent = (1 - self.smoothing_factor) * current_cent + self.smoothing_factor * prev_cent
                            smoothed_centroids_array[prev_idx] = smoothed_cent
                            
                            # Update frame_global_id_coords
                            fid = sorted_ids[current_idx]
                            centroid_rounded = (round(float(smoothed_cent[0]), 2), round(float(smoothed_cent[1]), 2))
                            frame_global_id_coords[fid] = centroid_rounded
                    
                    prev_smoothed_centroids = smoothed_centroids_array
                else:
                    # Cannot match, use current centroids
                    if len(current_centroids_array) > 0:
                        prev_smoothed_centroids = current_centroids_array.copy()
                    else:
                        prev_smoothed_centroids = None
            else:
                # No smoothing, prepare current centroids for next frame
                if len(frame_global_id_coords) > 0:
                    sorted_ids = sorted(frame_global_id_coords.keys())
                    prev_smoothed_centroids = np.array([frame_global_id_coords[fid] for fid in sorted_ids])
                else:
                    prev_smoothed_centroids = None
            
            frame_bev_coords[frame_idx] = frame_global_id_coords
            # Store in instance variable for later access
            self.frame_bev_coords[frame_idx] = frame_global_id_coords
            
            if (frame_idx + 1) % 10 == 0:
                print(f"  Processed {frame_idx + 1}/{num_frames} frames...")
        
        print(f"Multiview matching completed. Found {sum(len(m) for m in frame_multiview_maps.values())} mappings.")
        # Step 2: Temporal tracking of frame_global_ids across frames
        print("Step 2: Temporal tracking across frames...")
        
        # Prepare data for ObjectTracking: list_of_labels[frame_idx] = [(x, y), ...]
        # We need to map frame_global_id to index in the list
        # For offline tracking with fixed number of objects, ensure consistent ordering
        list_of_labels = []  # List of frames, each frame is list of (x, y) coordinates
        frame_global_id_to_idx = []  # For each frame: {frame_global_id: idx_in_list}
        
        # Store for later access
        self.frame_global_id_to_idx = frame_global_id_to_idx
        
        # Get expected number of objects from n_clusters
        expected_objects = self.n_clusters
        
        for frame_idx in range(num_frames):
            coords = frame_bev_coords.get(frame_idx, {})
            # Sort by frame_global_id to ensure consistent ordering
            sorted_ids = sorted(coords.keys())
            if sorted_ids:
                frame_labels = [coords[fid] for fid in sorted_ids]
            else:
                # Empty frame - no objects detected
                frame_labels = []
            
            # For offline tracking: if we have fewer objects than expected, 
            # trackpy will handle it (some objects might not be detected in this frame)
            list_of_labels.append(frame_labels)
            
            # Create mapping: frame_global_id -> idx
            frame_id_map = {fid: idx for idx, fid in enumerate(sorted_ids)}
            frame_global_id_to_idx.append(frame_id_map)
        
        # Use FixedIDTracker to ensure fixed number of IDs and minimal ID switches
        if expected_objects is not None and expected_objects > 0:
            tracker = FixedIDTracker(
                list_of_labels=list_of_labels,
                expected_objects=expected_objects,
                max_distance=self.max_distance_tracking
            )
            temporal_global_id_map = tracker.assign_ids()
        else:
            object_tracker = ObjectTracking(
                list_of_labels=list_of_labels,
                max_distance=self.max_distance_tracking
            )
            temporal_global_id_map = object_tracker.assign_global_id()
            
        
        # Store for later access
        self.temporal_global_id_map = temporal_global_id_map
        
        # Verify ID consistency: check if we have the expected number of unique IDs
        unique_ids = set(temporal_global_id_map.values())
        if expected_objects is not None:
            if len(unique_ids) > expected_objects:
                print(f"WARNING: Found {len(unique_ids)} unique IDs but expected {expected_objects} objects")
            elif len(unique_ids) < expected_objects:
                print(f"INFO: Found {len(unique_ids)} unique IDs (expected {expected_objects}, some objects may not appear in all frames)")
        
        print(f"Temporal tracking completed. Found {len(temporal_global_id_map)} temporal mappings with {len(unique_ids)} unique IDs.")
        
        # Step 3: Build final mapping: (frame_idx, camera_name, local_id) -> final_global_id
        print("Step 3: Building final global ID mapping...")
        
        self.final_global_id_map = {}
        missing_ids_count = 0
        
        for frame_idx in range(num_frames):
            multiview_map = frame_multiview_maps[frame_idx]
            frame_id_to_idx = frame_global_id_to_idx[frame_idx]
            
            for (camera_name, local_id), frame_global_id in multiview_map.items():
                # Get index in the list for this frame_global_id
                obj_idx = frame_id_to_idx.get(frame_global_id)
                if obj_idx is not None:
                    # Get final_global_id from temporal tracking
                    final_global_id = temporal_global_id_map.get((frame_idx, obj_idx), -1)
                    if final_global_id >= 0:
                        self.final_global_id_map[(frame_idx, camera_name, local_id)] = final_global_id
                    else:
                        # obj_idx not in temporal tracking - assign frame_global_id as fallback
                        # This can happen if the object was not tracked across frames
                        self.final_global_id_map[(frame_idx, camera_name, local_id)] = frame_global_id
                        missing_ids_count += 1
                else:
                    # frame_global_id not in frame_id_to_idx - this means no BEV coord was computed
                    # Use frame_global_id directly as final_global_id (fallback)
                    self.final_global_id_map[(frame_idx, camera_name, local_id)] = frame_global_id
                    missing_ids_count += 1
        
        if missing_ids_count > 0:
            print(f"WARNING: {missing_ids_count} boxes were assigned fallback IDs (not in temporal tracking)")
        
        # Verify all cameras were processed
        cameras_in_mapping = set()
        for (frame_idx, camera_name, local_id) in self.final_global_id_map.keys():
            cameras_in_mapping.add(camera_name)
        
        print(f"Final mapping completed. Total mappings: {len(self.final_global_id_map)}")
        print(f"Cameras processed in final mapping: {sorted(cameras_in_mapping)}")
        if len(cameras_in_mapping) != len(camera_names):
            missing_cams = set(camera_names) - cameras_in_mapping
            print(f"WARNING: Some cameras have no mappings: {missing_cams}")
        
        return self.final_global_id_map
    
    def get_global_id(self, frame_idx: int, camera_name: str, local_id: int) -> int:
        """Get final global ID for a local object"""
        return self.final_global_id_map.get((frame_idx, camera_name, local_id), -1)
    
    def get_all_global_ids(self) -> Dict[Tuple[int, str, int], int]:
        """Get all final global ID mappings"""
        return self.final_global_id_map.copy()
    
    def get_frame_bev_coords(self) -> Dict[int, Dict[int, Tuple[float, float]]]:
        """
        Get BEV coordinates (centroids) for each frame and frame_global_id.
        
        Returns:
            Dict mapping frame_idx -> {frame_global_id: (x, y)} where (x, y) are BEV coordinates
        """
        return self.frame_bev_coords.copy()
    
    def get_centroids_by_final_id(self) -> Dict[Tuple[int, int], Tuple[float, float]]:
        """
        Get centroids mapped by (frame_idx, final_global_id).
        
        Returns:
            Dict mapping (frame_idx, final_global_id) -> (x, y) BEV coordinates
        """
        result = {}
        
        if not hasattr(self, 'frame_global_id_to_idx') or not hasattr(self, 'temporal_global_id_map'):
            return result
        
        # Iterate through all frames and map frame_global_id centroids to final_global_id
        for frame_idx in range(len(self.frame_bev_coords)):
            frame_bev = self.frame_bev_coords.get(frame_idx, {})
            if not frame_bev:
                continue
                
            frame_id_to_idx = self.frame_global_id_to_idx[frame_idx] if frame_idx < len(self.frame_global_id_to_idx) else {}
            
            # Debug logging for first few frames
            if frame_idx < 3:
                print(f"  get_centroids_by_final_id: Frame {frame_idx}")
                print(f"    frame_bev keys (frame_global_ids): {sorted(frame_bev.keys())}")
                print(f"    frame_id_to_idx: {frame_id_to_idx}")
            
            # Map each frame_global_id centroid to its final_global_id
            for frame_global_id, centroid in frame_bev.items():
                obj_idx = frame_id_to_idx.get(frame_global_id)
                if obj_idx is not None:
                    final_global_id = self.temporal_global_id_map.get((frame_idx, obj_idx))
                    if final_global_id is not None:
                        # Store centroid for this final_global_id
                        # If multiple frame_global_ids map to same final_global_id (shouldn't happen),
                        # use the first one (they should all be the same after clustering)
                        key = (frame_idx, final_global_id)
                        if key not in result:
                            result[key] = centroid
                            
                            # Debug logging for first few frames
                            if frame_idx < 3:
                                print(f"    Mapped: frame_global_id={frame_global_id} -> obj_idx={obj_idx} -> final_global_id={final_global_id}, centroid={centroid}")
                    else:
                        # Fallback: use frame_global_id as final_global_id if temporal mapping not found
                        # This can happen if the detection was not tracked (e.g., too far from previous position)
                        final_global_id = frame_global_id
                        key = (frame_idx, final_global_id)
                        if key not in result:
                            result[key] = centroid
                        if frame_idx < 3:
                            print(f"    WARNING: frame_global_id={frame_global_id} -> obj_idx={obj_idx}, but no temporal mapping. Using frame_global_id={frame_global_id} as final_global_id")
                else:
                    # Fallback: use frame_global_id directly if not in frame_id_to_idx
                    # This should rarely happen, but provides a safety net
                    final_global_id = frame_global_id
                    key = (frame_idx, final_global_id)
                    if key not in result:
                        result[key] = centroid
                    if frame_idx < 3:
                        print(f"    WARNING: frame_global_id={frame_global_id} not found in frame_id_to_idx. Using frame_global_id={frame_global_id} as final_global_id")
        
        return result
