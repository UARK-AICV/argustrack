import cv2
import numpy as np
import os
from typing import Optional
from .linear_interpolation import get_dynamic_projection_point, estimate_distance_from_camera
from .calibration import CameraCalibration
from .constants import DEFAULT_BEV_X, DEFAULT_BEV_Y, DEFAULT_BEV_BOUNDS


class BEVProjector:
    """Project image points to BEV memory coordinates using CameraCalibration"""
    
    def __init__(self, calib_path, intrinsic_scale: float = 0.25, translation_scale: float = 100.0,
                 bev_x: int = DEFAULT_BEV_X, bev_y: int = DEFAULT_BEV_Y, 
                 bev_bounds: Optional[list] = None):
        """
        Initialize BEV projector
        
        Args:
            calib_path: Path to calibration folder containing intrinsic/extrinsic subfolders
            intrinsic_scale: Scale factor for intrinsic matrix (default 0.25 = 1/4)
            translation_scale: Scale factor for translation vector (default 100.0)
            bev_x: BEV grid X dimension (default from constants)
            bev_y: BEV grid Y dimension (default from constants)
            bev_bounds: BEV bounds [XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX] (default from constants)
        """
        self.calib_path = calib_path
        self.intrinsic_scale = intrinsic_scale
        self.translation_scale = translation_scale
        self.bev_x = bev_x
        self.bev_y = bev_y
        self.bev_bounds = bev_bounds if bev_bounds is not None else DEFAULT_BEV_BOUNDS
        self.cameras = {}  # Store CameraCalibration objects
        self.load_calibrations()
        
    def load_calibrations(self):
        """Load camera calibrations from XML files using CameraCalibration
        
        Supports two folder structures:
        1. Flat structure: calib_path/intrinsic/, calib_path/extrinsic/, calib_path/intrinsic_optimal/
        2. Per-camera structure: calib_path/cameraname/calibrations/intrinsic_optimal/, etc.
        """
        # Try flat structure first (for temp calibration folders from preprocessing)
        intrinsic_path = os.path.join(self.calib_path, 'intrinsic')
        extrinsic_path = os.path.join(self.calib_path, 'extrinsic')
        optimal_intrinsic_path = os.path.join(self.calib_path, 'intrinsic_optimal')
        
        if os.path.exists(intrinsic_path) and os.path.exists(extrinsic_path) and os.path.exists(optimal_intrinsic_path):
            # Flat structure found
            camera_names = []
            
            for filename in os.listdir(intrinsic_path):
                if filename.startswith('intr_') and filename.endswith('.xml'):
                    cam_name = filename.replace('intr_', '').replace('.xml', '')
                    camera_names.append(cam_name)
            
            print(f"Find {len(camera_names)} cameras (flat structure): {camera_names}")
            
            for cam_name in camera_names:
                intr_file = os.path.join(intrinsic_path, f'intr_{cam_name}.xml')
                extr_file = os.path.join(extrinsic_path, f'extr_{cam_name}.xml')
                optimal_intr_file = os.path.join(optimal_intrinsic_path, f'intr_{cam_name}.xml')
                
                # Load calibration using CameraCalibration class with provided scale factors
                calib = CameraCalibration.load_from_xml_files_scaled(
                    camera_id=cam_name,
                    intrinsic_path=intr_file, 
                    extrinsic_path=extr_file, 
                    optimal_intrinsic_path=optimal_intr_file,
                    intrinsic_scale=self.intrinsic_scale, 
                    translation_scale=self.translation_scale
                )
                if calib is not None:
                    self.cameras[cam_name] = calib
        else:
            # Try per-camera structure: calib_path/cameraname/calibrations/intrinsic_optimal/
            if not os.path.isdir(self.calib_path):
                print(f"Warning: Calibration path not found: {self.calib_path}")
                return
            
            # Scan for camera folders
            camera_folders = []
            for item in os.listdir(self.calib_path):
                item_path = os.path.join(self.calib_path, item)
                if os.path.isdir(item_path):
                    # Check if this folder contains calibrations subfolder
                    calib_subfolder = os.path.join(item_path, "calibrations")
                    if os.path.isdir(calib_subfolder):
                        camera_folders.append((item, calib_subfolder))
            
            if not camera_folders:
                print(f"Warning: No camera calibration folders found in: {self.calib_path}")
                return
            
            print(f"Found {len(camera_folders)} camera folders (per-camera structure)")
            
            for cam_folder_name, calib_subfolder in camera_folders:
                # Try to extract camera name from folder name (e.g., "3" -> "Camera3", "Camera3" -> "Camera3")
                import re
                match = re.search(r'(\d+)', cam_folder_name)
                if match:
                    cam_num = match.group(1)
                    cam_name = f"Camera{cam_num}"
                else:
                    # Use folder name as-is if it already looks like a camera name
                    cam_name = cam_folder_name if cam_folder_name.startswith("Camera") else f"Camera{cam_folder_name}"
                
                # Find intrinsic subfolder (try intrinsic_optimal first, then intrinsic_original, then intrinsic)
                intrinsic_subfolder = None
                for subfolder_name in ["intrinsic_optimal", "intrinsic_original", "intrinsic"]:
                    test_path = os.path.join(calib_subfolder, subfolder_name)
                    if os.path.isdir(test_path):
                        intrinsic_subfolder = test_path
                        break
                
                extrinsic_subfolder = os.path.join(calib_subfolder, "extrinsic")
                
                if not intrinsic_subfolder or not os.path.isdir(extrinsic_subfolder):
                    print(f"Warning: Calibration subfolders not found for {cam_name} in {calib_subfolder}")
                    continue
                
                # Try to find calibration files with different naming patterns
                intrinsic_files = []
                extrinsic_files = []
                
                # Pattern 1: intr_{cam_folder_name}.xml
                test_intr = os.path.join(intrinsic_subfolder, f"intr_{cam_folder_name}.xml")
                test_extr = os.path.join(extrinsic_subfolder, f"extr_{cam_folder_name}.xml")
                if os.path.exists(test_intr) and os.path.exists(test_extr):
                    intrinsic_files = [test_intr]
                    extrinsic_files = [test_extr]
                
                # Pattern 2: intr_{cam_name}.xml
                if not intrinsic_files:
                    test_intr = os.path.join(intrinsic_subfolder, f"intr_{cam_name}.xml")
                    test_extr = os.path.join(extrinsic_subfolder, f"extr_{cam_name}.xml")
                    if os.path.exists(test_intr) and os.path.exists(test_extr):
                        intrinsic_files = [test_intr]
                        extrinsic_files = [test_extr]
                
                # Pattern 3: intr_{cam_num}.xml (if we extracted a number)
                if not intrinsic_files and match:
                    test_intr = os.path.join(intrinsic_subfolder, f"intr_{cam_num}.xml")
                    test_extr = os.path.join(extrinsic_subfolder, f"extr_{cam_num}.xml")
                    if os.path.exists(test_intr) and os.path.exists(test_extr):
                        intrinsic_files = [test_intr]
                        extrinsic_files = [test_extr]
                
                # Pattern 4: any XML file in the folder
                if not intrinsic_files:
                    all_intrinsic = [f for f in os.listdir(intrinsic_subfolder) if f.endswith('.xml')]
                    all_extrinsic = [f for f in os.listdir(extrinsic_subfolder) if f.endswith('.xml')]
                    if all_intrinsic and all_extrinsic:
                        intrinsic_files = [os.path.join(intrinsic_subfolder, all_intrinsic[0])]
                        extrinsic_files = [os.path.join(extrinsic_subfolder, all_extrinsic[0])]
                
                if intrinsic_files and extrinsic_files:
                    intr_file = intrinsic_files[0]
                    extr_file = extrinsic_files[0]
                    optimal_intr_file = intr_file  # Use same file for optimal (since we're in intrinsic_optimal folder)
                    
                    # Load calibration
                    calib = CameraCalibration.load_from_xml_files_scaled(
                        camera_id=cam_name,
                        intrinsic_path=intr_file, 
                        extrinsic_path=extr_file, 
                        optimal_intrinsic_path=optimal_intr_file,
                        intrinsic_scale=self.intrinsic_scale, 
                        translation_scale=self.translation_scale
                    )
                    if calib is not None:
                        self.cameras[cam_name] = calib
                        print(f"Loaded calibration for {cam_name} from {calib_subfolder}")
            
        print(f"Success loaded {len(self.cameras)} cameras")
    
    def image_to_mem(self, cam_name: str, image_point: tuple, ground_z: float = 0.0) -> Optional[tuple]:
        """
        Project image point to BEV memory coordinates using CameraCalibration.project_2d_to_mem
        
        Args:
            cam_name: Camera name
            image_point: (x, y) pixel coordinates in image
            ground_z: Height of ground plane in world coordinates (default 0.0)
            
        Returns:
            (mem_x, mem_y) memory coordinates or None if projection fails
        """
        if cam_name not in self.cameras:
            raise ValueError(f"Camera {cam_name} does not exist!")
        
        calib = self.cameras[cam_name]
        
        # Use CameraCalibration.project_2d_to_mem method
        pixel_points = np.array([[image_point[0], image_point[1]]], dtype=np.float32)
        mem_points = calib.project_2d_to_mem(
            pixel_points,
            bev_x=self.bev_x,
            bev_y=self.bev_y,
            bev_bounds=self.bev_bounds,
            ground_z=ground_z,
            apply_undistortion=True
        )
        
        if mem_points is None or len(mem_points) == 0 or np.isnan(mem_points).any():
            return None
        
        mem_x, mem_y = mem_points[0]
        if np.isnan(mem_x) or np.isnan(mem_y):
            return None
        
        # Clamp mem coordinates to BEV bounds to ensure they're within valid range
        # This prevents points from being displayed incorrectly on BEV canvas
        x_min, x_max, y_min, y_max = self.bev_bounds[0], self.bev_bounds[1], self.bev_bounds[2], self.bev_bounds[3]
        
        # Check if coordinates are out of bounds (for debugging)
        if mem_x < x_min or mem_x > x_max or mem_y < y_min or mem_y > y_max:
            # Log warning but still clamp (don't return None)
            # This helps identify projection issues
            pass  # Commented out to avoid spam: logger.debug(f"BEV coords out of bounds: ({mem_x:.1f}, {mem_y:.1f}), bounds: [{x_min}, {x_max}] x [{y_min}, {y_max}]")
        
        mem_x = max(x_min, min(x_max, mem_x))
        mem_y = max(y_min, min(y_max, mem_y))
        
        return (float(mem_x), float(mem_y))
    
    def image_to_world(self, cam_name: str, image_point: tuple, z_world: float = 0.0) -> Optional[tuple]:
        """
        Project image point to world coordinates (legacy method, converts mem to world)
        
        Note: This method is kept for backward compatibility. It projects to mem coordinates
        first, then converts to world coordinates. For direct mem coordinates, use image_to_mem.
        
        Args:
            cam_name: Camera name
            image_point: (x, y) pixel coordinates in image
            z_world: Height in world coordinates (default 0.0)
            
        Returns:
            (X, Y, Z) world coordinates or None if projection fails
        """
        mem_coords = self.image_to_mem(cam_name, image_point, ground_z=z_world)
        if mem_coords is None:
            return None
        
        # Convert mem coordinates to world coordinates
        # For now, return mem coordinates as world coordinates (they're already in world grid)
        # This maintains backward compatibility
        mem_x, mem_y = mem_coords
        return (float(mem_x), float(mem_y), float(z_world))
    
    def get_calibration_info(self, camera_name):
        """Get calibration info for a camera (for compatibility with BBoxToBEVConverter)"""
        if camera_name not in self.cameras:
            return None
        
        calib = self.cameras[camera_name]
        R = calib.extrinsic[:3, :3]
        tvec = calib.extrinsic[:3, 3]
        # Convert R to rvec for compatibility
        rvec, _ = cv2.Rodrigues(R)
        
        return {
            'camera_matrix': calib.intrinsic,
            'R': R,
            'tvec': tvec,
            'rvec': rvec.flatten()
        }


class BBoxToBEVConverter:
    """Convert bounding boxes to BEV coordinates"""
    
    def __init__(self, calib_folder, average_object_height=2, min_dist=1.0, max_dist=10.0,
                 intrinsic_scale: float = 0.25, translation_scale: float = 100.0,
                 bev_x: int = DEFAULT_BEV_X, bev_y: int = DEFAULT_BEV_Y, 
                 bev_bounds: Optional[list] = None):
        self.calib_folder = calib_folder
        self.average_object_height = average_object_height
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.bev_projector = BEVProjector(
            calib_folder, 
            intrinsic_scale=intrinsic_scale, 
            translation_scale=translation_scale,
            bev_x=bev_x,
            bev_y=bev_y,
            bev_bounds=bev_bounds
        )
        self.calib_cache = {}
        # Cache image sizes to ensure consistency between frames
        self.image_size_cache = {}  # {camera_name: (width, height)}
    
    def set_image_size(self, camera_name: str, width: int, height: int):
        """Set image size for a camera (overrides estimation from camera_matrix)"""
        self.image_size_cache[camera_name] = (int(width), int(height))
    
    def _get_calibration(self, camera_name):
        """Get calibration info for a camera"""
        if camera_name not in self.calib_cache:
            calib_info = self.bev_projector.get_calibration_info(camera_name)
            if calib_info is None:
                raise ValueError(f"Camera {camera_name} not found in calibrations")
            self.calib_cache[camera_name] = calib_info
        return self.calib_cache[camera_name]
    
    def bbox_to_projection_point(self, camera_name, bbox):
        """Convert bbox to projection point in image coordinates"""
        calib_info = self._get_calibration(camera_name)
        
        # Round bbox to reduce sensitivity to small changes between frames.
        # This helps BEV projection remain stable when the bbox changes slightly.
        x_min, y_min, x_max, y_max = bbox
        # Round to 0.5 pixel to reduce noise while keeping accuracy
        x_min = round(float(x_min) * 2) / 2.0
        y_min = round(float(y_min) * 2) / 2.0
        x_max = round(float(x_max) * 2) / 2.0
        y_max = round(float(y_max) * 2) / 2.0
        bbox_rounded = [x_min, y_min, x_max, y_max]
        
        bbox_bottom_px = ((bbox_rounded[0] + bbox_rounded[2]) / 2, bbox_rounded[3])
        camera_extrinsics = {
            "R": calib_info["R"],
            "t": calib_info["tvec"]
        }
        camera_intrinsics = calib_info["camera_matrix"]
        distance = estimate_distance_from_camera(
            bbox_bottom_px, 
            camera_extrinsics, 
            camera_intrinsics, 
            ground_z=0
        )
        
        # Check for invalid distance
        if not np.isfinite(distance) or distance <= 0:
            return None
        
        # Round distance to reduce sensitivity.
        # Round to 0.1 m to make it more stable.
        distance = round(float(distance) * 10) / 10.0
        
        # Get image size from cache or estimate from camera_matrix.
        # Ensure the image size is consistent between frames for the same camera.
        if camera_name not in self.image_size_cache:
            fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
            cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
            # Estimate image size: typically cx, cy are near the center,
            # so width ≈ 2*cx and height ≈ 2*cy.
            # To be safe, fall back to defaults or calibration values.
            # If nothing is available, use standard size (960x540 as in test_process_label.py).
            img_width = max(int(2 * cx), 960)  # Default to 960 if estimate is too small
            img_height = max(int(2 * cy), 540)  # Default to 540 if estimate is too small
            # Round to ensure consistency
            img_width = int(round(img_width / 10.0) * 10)  # Round to 10 pixels
            img_height = int(round(img_height / 10.0) * 10)  # Round to 10 pixels
            self.image_size_cache[camera_name] = (img_width, img_height)
        else:
            img_width, img_height = self.image_size_cache[camera_name]
        
        projection_point, _ = get_dynamic_projection_point(
            bbox_rounded, 
            distance, 
            calib_info.get('rvec'), 
            self.min_dist, 
            self.max_dist,
            img_width=img_width,
            img_height=img_height
        )
        
        # Validate projection point
        if projection_point is None or not np.isfinite(projection_point).all():
            return None
        
        return projection_point
    
    def projection_point_to_bev(self, camera_name, projection_point, z_world=0.0):
        """Convert projection point to BEV memory coordinates"""
        # Do not round projection point to preserve accuracy and temporal consistency.
        # Rounding could cause BEV points to jump between frames.
        mem_coords = self.bev_projector.image_to_mem(
            camera_name, 
            projection_point, 
            ground_z=z_world
        )
        
        if mem_coords is not None:
            return mem_coords
        
        return None
    
    def bbox_to_bev(self, camera_name, bbox, z_world=0.0):
        """Convert bbox to BEV coordinates"""
        result = {
            'bev_coords': None,
            'projection_point': None,
            'success': False,
            'camera': camera_name
        }
        
        try:
            projection_point = self.bbox_to_projection_point(camera_name, bbox)
            if projection_point is None:
                return result
            result['projection_point'] = projection_point
            bev_coords = self.projection_point_to_bev(camera_name, projection_point, z_world)
            
            if bev_coords is None:
                return result
            
            result['bev_coords'] = bev_coords
            result['success'] = True
            
        except Exception as e:
            print(f"Error {camera_name}: {e}")
            result['error'] = str(e)
        
        return result
    
    def batch_bbox_to_bev(self, bbox_dict, z_world=0.0):
        """Convert multiple bboxes to BEV coordinates"""
        results = {}
        
        for camera_name, bbox in bbox_dict.items():
            results[camera_name] = self.bbox_to_bev(camera_name, bbox, z_world)
        
        return results


def convert_bbox_to_bev(camera_name, bbox, calib_folder, 
                        average_object_height=0.3, min_dist=1.0, max_dist=10.0, 
                        z_world=0.0, intrinsic_scale: float = 0.25, 
                        translation_scale: float = 100.0):
    """Standalone function to convert bbox to BEV coordinates"""
    try:
        bev_projector = BEVProjector(
            calib_folder, 
            intrinsic_scale=intrinsic_scale, 
            translation_scale=translation_scale
        )
        calib_info = bev_projector.get_calibration_info(camera_name)
        
        if calib_info is None:
            return None
        
        bbox_bottom_px = ((bbox[0] + bbox[2]) / 2, bbox[3])
        camera_extrinsics = {
            "R": calib_info["R"],
            "t": calib_info["tvec"]
        }
        camera_intrinsics = calib_info["camera_matrix"]
        distance = estimate_distance_from_camera(
            bbox_bottom_px, 
            camera_extrinsics, 
            camera_intrinsics, 
            ground_z=0
        )
        projection_point, _ = get_dynamic_projection_point(
            bbox, 
            distance, 
            calib_info.get('rvec'), 
            min_dist, 
            max_dist
        )
        
        if projection_point is None:
            return None
        
        mem_coords = bev_projector.image_to_mem(camera_name, projection_point, ground_z=z_world)
        
        if mem_coords is not None:
            return mem_coords
        
        return None
        
    except Exception as e:
        print(f"Error: {e}")
        return None
