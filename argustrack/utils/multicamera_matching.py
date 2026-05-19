from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.cluster import KMeans
from .project_bev import BBoxToBEVConverter
from ultralytics import YOLO


class MultiCameraTracking:
    """Multi-camera object mapping (multiview matching only, no temporal tracking)"""
    
    def __init__(self, calib_folder: str, model_name: str = None, model_path: str = None, 
                 data_path: str = None, max_distance: float = 2.0, n_clusters: int = None,
                 intrinsic_scale: float = 0.25, translation_scale: float = 100.0,
                 bev_x: int = None, bev_y: int = None, bev_bounds: list = None):
        """
        Args:
            calib_folder: Path to calibration folder containing intrinsic/extrinsic
            model_name: Model name for detection (optional)
            model_path: Path to model file (optional)
            data_path: Path to data (optional)
            max_distance: Maximum distance in BEV space for clustering objects (meters) - deprecated, not used with KMeans
            n_clusters: Number of clusters for KMeans. If None, will be auto-determined from number of detections
            intrinsic_scale: Scale factor for intrinsic matrix (default 0.25 = 1/4)
            translation_scale: Scale factor for translation vector (default 100.0)
            bev_x: BEV grid X dimension (default from constants)
            bev_y: BEV grid Y dimension (default from constants)
            bev_bounds: BEV bounds [XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX] (default from constants)
        """
        from argustrack.utils.constants import DEFAULT_INTRINSIC_SCALE, DEFAULT_TRANSLATION_SCALE
        from argustrack.utils.constants import DEFAULT_BEV_X, DEFAULT_BEV_Y, DEFAULT_BEV_BOUNDS
        
        self.calib_folder = calib_folder
        self.model_name = model_name
        self.model_path = model_path
        self.data_path = data_path
        self.max_distance = max_distance
        self.n_clusters = n_clusters
        self.model = None
        
        # Use provided scale factors or defaults from constants
        intrinsic_scale = intrinsic_scale if intrinsic_scale is not None else DEFAULT_INTRINSIC_SCALE
        translation_scale = translation_scale if translation_scale is not None else DEFAULT_TRANSLATION_SCALE
        bev_x = bev_x if bev_x is not None else DEFAULT_BEV_X
        bev_y = bev_y if bev_y is not None else DEFAULT_BEV_Y
        bev_bounds = bev_bounds if bev_bounds is not None else DEFAULT_BEV_BOUNDS.copy()
        
        # Initialize BEV converter with scale factors
        self.bev_converter = BBoxToBEVConverter(
            calib_folder,
            intrinsic_scale=intrinsic_scale,
            translation_scale=translation_scale,
            bev_x=bev_x,
            bev_y=bev_y,
            bev_bounds=bev_bounds
        )
        
        # Mapping results (for single frame)
        self.global_id_map = {}  # {(camera_name, local_id): global_id}
        self.local_id_map = {}   # {(camera_name, global_id): local_id}
        self.bev_coords = {}     # {(camera_name, local_id): (x, y)}
        self.cluster_centroids = {}  # {cluster_id: (x, y)} - KMeans centroids for each cluster
        
    def load_model(self):
        """Load detection model"""
        if self.model_name == "yolo":
            self.model = YOLO(self.model_path)
        else:
            raise ValueError(f"Model {self.model_name} not supported")

    def load_data(self):
        """Load data - to be implemented based on data format"""
        pass
    
    def project_bev(self, detections: Dict[str, List[List[float]]]) -> Dict[str, List[Optional[Tuple[float, float]]]]:
        """
        Project bounding boxes from all cameras to BEV coordinates
        
        Args:
            detections: Dict mapping camera_name -> list of bboxes [x_min, y_min, x_max, y_max]
            
        Returns:
            Dict mapping camera_name -> list of BEV coordinates (x, y) or None
        """
        bev_results = {}
        
        # Process ALL cameras in detections - ensure none are skipped
        for camera_name, bboxes in detections.items():
            bev_coords_list = []
            for local_id, bbox in enumerate(bboxes):
                result = self.bev_converter.bbox_to_bev(camera_name, bbox, z_world=0.0)
                if result['success']:
                    bev_coords_list.append(result['bev_coords'])
                else:
                    bev_coords_list.append(None)
            bev_results[camera_name] = bev_coords_list
        
        # Verify all cameras were processed
        if len(bev_results) != len(detections):
            missing_cams = set(detections.keys()) - set(bev_results.keys())
            print(f"WARNING: Some cameras were not processed in project_bev: {missing_cams}")
            # Add missing cameras with empty results
            for cam_name in missing_cams:
                bev_results[cam_name] = []
        
        return bev_results
    
    def _cluster_multiview_detections(
            self,
            bev_coords: List[Tuple[float, float, str, int]],
            prev_centroids: Optional[np.ndarray] = None,
            max_centroid_movement: float = 5.0
        ) -> Dict[int, List[Tuple[str, int]]]:
        """
        Cluster BEV coordinates with a hard constraint:
        - **Ràng buộc quan trọng**: Một cluster **không được** chứa 2 detections đến từ **cùng một camera**.
        - Dùng KMeans để lấy các centroid ban đầu, sau đó gán lại greedily theo khoảng cách,
          đảm bảo mỗi (cluster, camera) xuất hiện tối đa 1 lần.
        - Nếu một detection không thể được gán vào bất kỳ cluster KMeans nào mà vẫn giữ được ràng buộc,
          nó sẽ được đưa vào một cluster mới (singleton cluster), vẫn đảm bảo ràng buộc theo camera.
        
        Args:
            bev_coords: List of (x, y, camera_name, local_id)
            prev_centroids: Previous frame's cluster centers for warm start (shape: [n_clusters, 2])
            max_centroid_movement: Maximum allowed movement of cluster centers in meters (for stability)
        """
        if not bev_coords:
            return {}

        # Prepare data
        all_coords = [(x, y) for x, y, _, _ in bev_coords]
        coord_to_camera = [(cam, det_idx) for _, _, cam, det_idx in bev_coords]
        coords_array = np.array(all_coords, dtype=np.float32)

        n_points = len(coords_array)
        all_cams = {cam for cam, _ in coord_to_camera}
        size_limit = max(1, len(all_cams))  # maximum detections per cluster is #cams

        # Determine initial KMeans cluster count
        if self.n_clusters is None:
            # Rough estimate: each object appears in approximately 3 cameras
            estimated_clusters = max(1, n_points // 3)
            n_clusters = min(estimated_clusters, n_points)
        else:
            n_clusters = min(max(1, self.n_clusters), n_points)

        if n_clusters <= 0:
            return {}

        # KMeans to get centroids with warm start from previous frame
        if prev_centroids is not None and len(prev_centroids) == n_clusters:
            # Use previous centroids as initialization
            kmeans = KMeans(n_clusters=n_clusters, n_init=1, init=prev_centroids, random_state=42)
            kmeans.fit(coords_array)
            centroids = kmeans.cluster_centers_
            
            # Apply constraint: limit centroid movement distance
            for i in range(len(centroids)):
                if i < len(prev_centroids):
                    dist = np.linalg.norm(centroids[i] - prev_centroids[i])
                    if dist > max_centroid_movement:
                        # If moved too far, keep old position and adjust slightly
                        direction = (centroids[i] - prev_centroids[i]) / dist
                        centroids[i] = prev_centroids[i] + direction * max_centroid_movement
        else:
            # No previous centroids or count mismatch, use random init
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            kmeans.fit(coords_array)
            centroids = kmeans.cluster_centers_

        # Calculate distances to centroids
        dists = np.linalg.norm(coords_array[:, None, :] - centroids[None, :, :], axis=2)

        # Detection order: closest to centroid first
        nearest_centroid_dist = np.min(dists, axis=1)
        detection_order = list(np.argsort(nearest_centroid_dist))

        clusters: Dict[int, List[int]] = {i: [] for i in range(n_clusters)}
        cluster_cameras: Dict[int, set] = {i: set() for i in range(n_clusters)}
        unassigned: List[int] = []

        # Assign: nearest cluster, NO same camera, not exceeding size_limit
        for idx in detection_order:
            cam_name, _ = coord_to_camera[idx]
            centroid_candidates = list(np.argsort(dists[idx]))
            assigned = False
            for c in centroid_candidates:
                if len(clusters[c]) >= size_limit:
                    continue
                if cam_name in cluster_cameras[c]:
                    continue  # constraint: 1 cluster cannot contain 2 objects from same camera
                clusters[c].append(idx)
                cluster_cameras[c].add(cam_name)
                assigned = True
                break

            if not assigned:
                unassigned.append(idx)

        # For unassigned detections: try to assign to existing clusters
        # CRITICAL: If n_clusters is set, we must NOT create new clusters beyond n_clusters
        # Instead, try to assign unassigned detections to existing clusters that have space
        for idx in unassigned:
            cam_name, _ = coord_to_camera[idx]
            assigned = False
            # Try to find a cluster that:
            # 1. Has space (not at size_limit)
            # 2. Doesn't already have this camera
            # 3. Is closest to this detection
            best_cluster = None
            min_dist = float('inf')
            for c in range(n_clusters):
                if len(clusters[c]) >= size_limit:
                    continue
                if cam_name in cluster_cameras[c]:
                    continue
                # Check distance to cluster centroid
                if len(clusters[c]) > 0:
                    # Use centroid of existing points in cluster
                    cluster_points = [all_coords[i] for i in clusters[c]]
                    cluster_center = np.mean(np.array(cluster_points), axis=0)
                    detection_point = coords_array[idx]
                    dist_to_cluster = np.linalg.norm(detection_point - cluster_center)
                    if dist_to_cluster < min_dist:
                        min_dist = dist_to_cluster
                        best_cluster = c
                else:
                    # Empty cluster, use centroid from KMeans
                    if c < len(centroids):
                        detection_point = coords_array[idx]
                        dist_to_cluster = np.linalg.norm(detection_point - centroids[c])
                        if dist_to_cluster < min_dist:
                            min_dist = dist_to_cluster
                            best_cluster = c
            
            # Assign to best cluster if found
            if best_cluster is not None:
                clusters[best_cluster].append(idx)
                cluster_cameras[best_cluster].add(cam_name)
                assigned = True
            # If still not assigned and n_clusters is set, we must drop this detection
            # to maintain the fixed number of clusters
            if not assigned:
                if self.n_clusters is not None:
                    # Drop this detection to maintain n_clusters constraint
                    print(f"    WARNING: Dropping unassigned detection from {cam_name} to maintain n_clusters={self.n_clusters}")
                else:
                    # If n_clusters is None, create new cluster (original behavior)
                    next_cluster_id = len(clusters)
                    clusters[next_cluster_id] = [idx]
                    cluster_cameras[next_cluster_id] = {cam_name}
                    # Use the detection point itself as representative footpoint
                    detection_point = coords_array[idx]
                    self.cluster_centroids[next_cluster_id] = (float(detection_point[0]), float(detection_point[1]))

        # Build final result: {cluster_id: [(camera_name, local_id), ...]}
        result: Dict[int, List[Tuple[str, int]]] = {}
        # Store centroids for each cluster (will be computed from actual assigned points)
        # Initialize with any existing centroids (e.g., from new clusters created above)
        cluster_centroids_temp = self.cluster_centroids.copy() if hasattr(self, 'cluster_centroids') else {}
        self.cluster_centroids = {}
        
        for cid, point_indices in clusters.items():
            if not point_indices:
                continue
            members: List[Tuple[str, int]] = []
            for idx in point_indices:
                cam_name, det_idx = coord_to_camera[idx]
                members.append((cam_name, det_idx))
            result[cid] = members
            
            # Instead of computing centroid, select a representative footpoint from the cluster
            # Choose the point closest to the cluster centroid (mean of all points)
            cluster_points = [all_coords[i] for i in point_indices]
            if cluster_points:
                points_array = np.array(cluster_points, dtype=np.float32)
                # Compute centroid first to find the representative point
                actual_centroid = np.mean(points_array, axis=0)
                
                # Find the point closest to centroid (most representative footpoint)
                distances = np.linalg.norm(points_array - actual_centroid, axis=1)
                closest_idx = np.argmin(distances)
                representative_point = points_array[closest_idx]
                
                # Store the representative footpoint instead of centroid
                self.cluster_centroids[cid] = (float(representative_point[0]), float(representative_point[1]))
            elif cid in cluster_centroids_temp:
                # Use existing point if cluster has no points (shouldn't happen, but safety)
                self.cluster_centroids[cid] = cluster_centroids_temp[cid]

        return result

    
    def match_multiview(self, detections: Dict[str, List[List[float]]], 
                       prev_centroids: Optional[np.ndarray] = None,
                       max_centroid_movement: float = 5.0) -> Dict[Tuple[str, int], int]:
        """
        Match objects across multiple cameras in a single frame using K-means clustering
        
        Args:
            detections: Dict mapping camera_name -> list of bboxes [x_min, y_min, x_max, y_max]
            prev_centroids: Previous frame's cluster centers for warm start (shape: [n_clusters, 2])
            max_centroid_movement: Maximum allowed movement of cluster centers in meters (for stability)
            
        Returns:
            Dict mapping (camera_name, local_id) -> global_id
        """
        if not detections:
            return {}
        
        # Reset mappings for new frame
        self.global_id_map = {}
        self.local_id_map = {}
        self.bev_coords = {}
        self.cluster_centroids = {}
        
        # Project to BEV
        bev_results = self.project_bev(detections)
        
        # Collect valid BEV coordinates
        bev_coords_list = []  # List of (x, y, camera_name, local_id)
        invalid_detections = []  # List of (camera_name, local_id) with invalid BEV
        
        for camera_name, bev_coords in bev_results.items():
            for local_id, bev_coord in enumerate(bev_coords):
                if bev_coord is not None:
                    x, y = bev_coord
                    # Check for NaN or Inf values
                    if np.isfinite(x) and np.isfinite(y):
                        bev_coords_list.append((x, y, camera_name, local_id))
                        # Store for later lookup
                        self.bev_coords[(camera_name, local_id)] = (x, y)
                    else:
                        # Invalid BEV coordinate
                        invalid_detections.append((camera_name, local_id))
                else:
                    # BEV projection failed
                    invalid_detections.append((camera_name, local_id))
        
        # Cluster BEV coordinates
        if bev_coords_list:
            # Debug: Log BEV coordinates before clustering
            if len(bev_coords_list) > 0 and len(bev_coords_list) <= 20:  # Only log for small number of detections
                print(f"    Clustering {len(bev_coords_list)} BEV coordinates from {len(detections)} cameras")
                cam_counts = {}
                for x, y, cam_name, local_id in bev_coords_list:
                    if cam_name not in cam_counts:
                        cam_counts[cam_name] = 0
                    cam_counts[cam_name] += 1
                print(f"      BEV coords per camera: {cam_counts}")
            
            clusters = self._cluster_multiview_detections(bev_coords_list, prev_centroids, max_centroid_movement)
            
            # Debug: Log clustering results
            if len(clusters) > 0:
                print(f"    Clustering created {len(clusters)} clusters")
                for cluster_id in sorted(clusters.keys()):
                    members = clusters[cluster_id]
                    if members:
                        cams_in_cluster = [cam for cam, _ in members]
                        print(f"      Cluster {cluster_id}: {len(members)} detections from {len(set(cams_in_cluster))} cameras: {set(cams_in_cluster)}")
            
            next_global_id = 0
            # Map cluster_id to global_id for centroid lookup
            cluster_to_global_id = {}  # {cluster_id: global_id}

            # Assign consecutive global_id to each cluster
            for cluster_id in sorted(clusters.keys()):
                members = clusters[cluster_id]
                if not members:
                    continue
                global_id = next_global_id
                next_global_id += 1
                cluster_to_global_id[cluster_id] = global_id
                for camera_name, local_id in members:
                    self.global_id_map[(camera_name, local_id)] = global_id
                    self.local_id_map[(camera_name, global_id)] = local_id
            
            # Map cluster representative footpoints from cluster_id to global_id
            # This ensures we can look up footpoints by global_id later
            global_id_footpoints = {}  # {global_id: (x, y)}
            for cluster_id, footpoint in self.cluster_centroids.items():
                if cluster_id in cluster_to_global_id:
                    global_id = cluster_to_global_id[cluster_id]
                    global_id_footpoints[global_id] = footpoint
            # Replace cluster_centroids with global_id_footpoints (keeping same variable name for compatibility)
            self.cluster_centroids = global_id_footpoints
        else:
            next_global_id = 0
            print(f"    WARNING: No valid BEV coordinates to cluster!")
        
        # Assign separate IDs to invalid detections (those with failed BEV projection)
        # Each invalid detection gets its own unique ID after reserved cluster IDs
        for camera_name, local_id in invalid_detections:
            self.global_id_map[(camera_name, local_id)] = next_global_id
            next_global_id += 1

        if not self.global_id_map:
            # Don't print warning here - it will be logged at higher level if needed
            return {}
        
        return self.global_id_map
    
    def get_global_id(self, camera_name: str, local_id: int) -> int:
        """Get global ID for a local object"""
        return self.global_id_map.get((camera_name, local_id), -1)
    
    def get_local_id(self, camera_name: str, global_id: int) -> Optional[int]:
        """Get local ID for a global object in a specific camera"""
        return self.local_id_map.get((camera_name, global_id))
    
    def get_bev_coords(self, camera_name: str, local_id: int) -> Optional[Tuple[float, float]]:
        """Get BEV coordinates for an object"""
        return self.bev_coords.get((camera_name, local_id))
    
    def get_cluster_centroids(self) -> Optional[np.ndarray]:
        """
        Get current frame's cluster representative footpoints after clustering.
        Returns footpoints as numpy array of shape [n_clusters, 2] or None if not available.
        Uses stored representative footpoints (one per cluster, closest to centroid).
        """
        if not self.cluster_centroids:
            return None
        
        # Get footpoints sorted by cluster ID
        sorted_ids = sorted(self.cluster_centroids.keys())
        footpoints = [self.cluster_centroids[cid] for cid in sorted_ids]
        
        if not footpoints:
            return None
        
        return np.array(footpoints, dtype=np.float32)
    
    def get_cluster_centroid(self, cluster_id: int) -> Optional[Tuple[float, float]]:
        """
        Get representative footpoint for a specific cluster ID.
        Returns (x, y) or None if cluster not found.
        Note: Despite the name, this returns a footpoint (not centroid) - one point from the cluster.
        """
        return self.cluster_centroids.get(cluster_id)
    
    def map_to_local_ids(self, camera_name: str, global_ids: List[int]) -> Dict[int, int]:
        """
        Map global IDs to local IDs for a specific camera
        
        Args:
            camera_name: Camera name
            global_ids: List of global IDs to map
            
        Returns:
            Dict mapping global_id -> local_id
        """
        result = {}
        for global_id in global_ids:
            local_id = self.get_local_id(camera_name, global_id)
            if local_id is not None:
                result[global_id] = local_id
        return result