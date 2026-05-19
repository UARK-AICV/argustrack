"""Utilities for loading and using camera calibrations for 3D projection"""

import cv2
import torch
import json
import numpy as np
import os.path as osp
from loguru import logger

import xml.etree.ElementTree as ET
from typing import Optional, Dict, Any, List

from argustrack.utils.vox_utils import vox
from argustrack.utils.constants import get_worldgrid2worldcoord_torch, DEFAULT_BEV_BOUNDS
from argustrack.utils.constants import DEFAULT_BEV_X, DEFAULT_BEV_Y, IMAGE_WIDTH, IMAGE_HEIGHT

class CameraCalibration:
    """Represents camera calibration data"""
    
    def __init__(
        self, 
        camera_id: str,
        intrinsic: np.ndarray, 
        extrinsic: np.ndarray,
        optimal_intrinsic: Optional[np.ndarray] = None,
        distortion: Optional[np.ndarray] = None,
        bev_bounds: Optional[List] = None,
    ):
        """
        Initialize camera calibration
        
        Args:
            camera_id: camera id
            intrinsic: 3x3 camera intrinsic matrix
            extrinsic: 4x4 camera extrinsic matrix (world to camera)
            optimal_intrinsic: 3x3 optimal intrinsic matrix (optional)
            distortion: distortion coefficients (optional)
            bev_bounds: BEV bounds [XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX] (optional)
        """
        self.camera_id = camera_id
        self.intrinsic = np.array(intrinsic, dtype=np.float32)
        self.extrinsic = np.array(extrinsic, dtype=np.float32)
        self.distortion = np.array(distortion, dtype=np.float32) if distortion is not None else np.zeros(5, dtype=np.float32)
        self.optimal_intrinsic = np.array(optimal_intrinsic, dtype=np.float32) if optimal_intrinsic is not None else np.eye(3, dtype=np.float32)
        # Validate matrices
        if self.intrinsic.shape != (3, 3):
            raise ValueError(f"Intrinsic matrix must be 3x3, got {self.intrinsic.shape}")
        if self.extrinsic.shape != (4, 4):
            raise ValueError(f"Extrinsic matrix must be 4x4, got {self.extrinsic.shape}")
    
    @classmethod
    def load_from_xml_files(cls, intrinsic_path: str, extrinsic_path: str) -> Optional['CameraCalibration']:
        """Load calibration from OpenCV XML files"""
        try:
            # Load intrinsic
            intrinsic = cls._load_intrinsic_xml(intrinsic_path)
            if intrinsic is None:
                return None
            
            # Load extrinsic
            extrinsic = cls._load_extrinsic_xml(extrinsic_path)
            if extrinsic is None:
                return None
            
            # Load distortion (from intrinsic file)
            distortion = cls._load_distortion_xml(intrinsic_path)
            
            return cls(intrinsic, extrinsic, distortion)
        except Exception as e:
            logger.error(f"Failed to load calibration from XML files: {e}")
            return None
    
    @classmethod
    def load_from_xml_files_scaled(
        cls, 
        camera_id: str,
        intrinsic_path: str, 
        extrinsic_path: str,
        optimal_intrinsic_path: str,
        intrinsic_scale: float = 1.0,
        translation_scale: float = 1.0
    ) -> Optional['CameraCalibration']:
        """
        Load calibration from OpenCV XML files with scaling applied.
        
        Args:
            camera_id: camera id
            intrinsic_path: Path to intrinsic XML file
            extrinsic_path: Path to extrinsic XML file
            intrinsic_scale: Scale factor for intrinsic matrix (fx, fy, cx, cy)
            translation_scale: Scale factor for translation vector in extrinsic
            optimal_intrinsic_path: Path to optimal intrinsic XML file
            
        Returns:
            CameraCalibration with scaled parameters
        """
        try:
            # Load intrinsic
            intrinsic = cls._load_intrinsic_xml(intrinsic_path)
            if intrinsic is None:
                return None
            
            # Load extrinsic
            extrinsic = cls._load_extrinsic_xml(extrinsic_path)
            if extrinsic is None:
                return None
            
            # Load distortion (from intrinsic file)
            distortion = cls._load_distortion_xml(intrinsic_path)
            
            # Apply intrinsic scale
            scaled_intrinsic = cls._scale_intrinsic(intrinsic, intrinsic_scale)
            
            # Apply translation scale
            scaled_extrinsic = cls._scale_extrinsic_translation(extrinsic, translation_scale)
            
            # Load optimal intrinsic
            optimal_intrinsic = cls._load_intrinsic_xml(optimal_intrinsic_path)
            optimal_intrinsic = cls._scale_intrinsic(optimal_intrinsic, intrinsic_scale)
            
            return cls(camera_id, scaled_intrinsic, scaled_extrinsic, optimal_intrinsic, distortion)
        except Exception as e:
            logger.error(f"Failed to load scaled calibration from XML files: {e}")
            return None
    
    @staticmethod
    def _scale_intrinsic(K: np.ndarray, scale: float) -> np.ndarray:
        """Scale intrinsic matrix by given factor"""
        K_scaled = K.copy()
        K_scaled[0, 0] *= scale  # fx
        K_scaled[1, 1] *= scale  # fy
        K_scaled[0, 2] *= scale  # cx
        K_scaled[1, 2] *= scale  # cy
        return K_scaled
    
    @staticmethod
    def _scale_extrinsic_translation(extrinsic: np.ndarray, scale: float) -> np.ndarray:
        """Scale translation vector in extrinsic matrix"""
        scaled = extrinsic.copy()
        scaled[:3, 3] *= scale  # Scale translation vector
        return scaled
    
    @staticmethod
    def _load_intrinsic_xml(filepath: str) -> Optional[np.ndarray]:
        """Load intrinsic matrix from OpenCV XML file"""
        if not osp.exists(filepath):
            logger.warning(f"Intrinsic file not found: {filepath}")
            return None
        
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            # Find camera_matrix
            camera_matrix_elem = root.find('camera_matrix')
            if camera_matrix_elem is None:
                logger.error(f"No camera_matrix found in {filepath}")
                return None
            
            # Parse matrix data
            rows = int(camera_matrix_elem.find('rows').text)
            cols = int(camera_matrix_elem.find('cols').text)
            data_str = camera_matrix_elem.find('data').text
            data = np.array([float(x) for x in data_str.split()], dtype=np.float32)
            
            # Reshape to matrix
            matrix = data.reshape(rows, cols)
            return matrix
        except Exception as e:
            logger.error(f"Failed to parse intrinsic XML {filepath}: {e}")
            return None
    
    @staticmethod
    def _load_distortion_xml(filepath: str) -> Optional[np.ndarray]:
        """Load distortion coefficients from OpenCV XML file"""
        if not osp.exists(filepath):
            return None
        
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            # Find distortion_coefficients
            dist_elem = root.find('distortion_coefficients')
            if dist_elem is None:
                return np.zeros(5, dtype=np.float32)
            
            # Parse distortion data
            rows = int(dist_elem.find('rows').text)
            data_str = dist_elem.find('data').text
            data = np.array([float(x) for x in data_str.split()], dtype=np.float32)
            
            return data
        except Exception as e:
            logger.warning(f"Failed to parse distortion from {filepath}: {e}")
            return np.zeros(5, dtype=np.float32)
    
    @staticmethod
    def _load_extrinsic_xml(filepath: str) -> Optional[np.ndarray]:
        """Load extrinsic matrix from OpenCV XML file (rvec + tvec -> 4x4 matrix)"""
        if not osp.exists(filepath):
            logger.warning(f"Extrinsic file not found: {filepath}")
            return None
        
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            # Find rvec and tvec
            rvec_elem = root.find('rvec')
            tvec_elem = root.find('tvec')
            
            if rvec_elem is None or tvec_elem is None:
                logger.error(f"Missing rvec or tvec in {filepath}")
                return None
            
            # Parse rvec (rotation vector)
            rvec_data_str = rvec_elem.find('data').text
            rvec = np.array([float(x) for x in rvec_data_str.split()], dtype=np.float32)
            
            # Parse tvec (translation vector)
            tvec_data_str = tvec_elem.find('data').text
            tvec = np.array([float(x) for x in tvec_data_str.split()], dtype=np.float32)
            
            # Convert rotation vector to rotation matrix using Rodrigues
            if cv2 is None:
                logger.error("OpenCV is required for Rodrigues conversion")
                return None
            R, _ = cv2.Rodrigues(rvec)
            
            # Build 4x4 extrinsic matrix (world to camera)
            extrinsic = np.eye(4, dtype=np.float32)
            extrinsic[:3, :3] = R
            extrinsic[:3, 3] = tvec
            
            return extrinsic
        except Exception as e:
            logger.error(f"Failed to parse extrinsic XML {filepath}: {e}")
            return None
    
    @classmethod
    def load_from_file(cls, filepath: str) -> Optional['CameraCalibration']:
        """Load calibration from JSON file (legacy support)"""
        if not osp.exists(filepath):
            logger.warning(f"Calibration file not found: {filepath}")
            return None
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            intrinsic = np.array(data.get("intrinsic", np.eye(3)), dtype=np.float32)
            extrinsic = np.array(data.get("extrinsic", np.eye(4)), dtype=np.float32)
            distortion = np.array(data.get("distortion", [0, 0, 0, 0, 0]), dtype=np.float32)
            
            return cls(intrinsic, extrinsic, distortion)
        except Exception as e:
            logger.error(f"Failed to load calibration from {filepath}: {e}")
            return None

    def _merge_intrinsics_4x4(self, K_3x3: np.ndarray) -> np.ndarray:
        """Convert 3x3 intrinsic to 4x4 format"""
        K = np.zeros((4, 4), dtype=K_3x3.dtype)
        K[0, 0] = K_3x3[0, 0]  # fx
        K[1, 1] = K_3x3[1, 1]  # fy
        K[0, 2] = K_3x3[0, 2]  # cx
        K[1, 2] = K_3x3[1, 2]  # cy
        K[2, 2] = 1.0
        K[3, 3] = 1.0
        return K

    def _get_undistorted_img(self, img: np.ndarray) -> np.ndarray:
        """Get undistorted image"""
        return cv2.undistort(img, self.intrinsic, self.distortion, None, self.optimal_intrinsic)

    def project_3d_to_2d(
        self, 
        points_3d: np.ndarray, 
        apply_distortion: bool = True,
        from_mem: bool = False,
        bev_x: int = 1200,
        bev_y: int = 800,
        bev_bounds: Optional[List] = None,
        undistorted_img: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Project 3D points to 2D image coordinates.
        
        If from_mem=True, applies full transformation chain:
            mem -> ref (ref_T_mem) -> world (worldgrid2worldcoord) -> camera (extrinsic) 
            -> pixel (intrinsic) -> distorted
        
        If from_mem=False, applies standard projection:
            world -> camera (extrinsic) -> pixel (intrinsic) -> distorted
        
        Args:
            points_3d: Nx3 array of points (mem coordinates if from_mem=True, else world coordinates)
            apply_distortion: Whether to apply lens distortion
            from_mem: If True, input is memory coordinates; apply ref_T_mem and worldgrid2worldcoord first
            bev_x: BEV grid X dimension (only used if from_mem=True)
            bev_y: BEV grid Y dimension (only used if from_mem=True)
            bev_bounds: BEV bounds (only used if from_mem=True)
            
        Returns:
            Nx2 array of 2D image coordinates, NaN for points behind camera.
        """
        if points_3d.shape[1] != 3:
            raise ValueError(f"points_3d must be Nx3, got {points_3d.shape}")
        
        # Ensure float32
        points_3d = np.asarray(points_3d, dtype=np.float32)
        n_points = points_3d.shape[0]
        
        device = 'cpu'
        B = 1
        X, Y, Z = bev_x, bev_y, 2
        worldgrid2worldcoord = get_worldgrid2worldcoord_torch(device)  # 4 x 4 (homogeneous)
        # Convert Tensor to numpy array to avoid type mismatch
        if isinstance(worldgrid2worldcoord, torch.Tensor):
            worldgrid2worldcoord = worldgrid2worldcoord.cpu().numpy()
        
        # worldgrid2worldcoord is 4x4, use it directly as 4x4
        ref_T_global_4x4 = np.linalg.inv(worldgrid2worldcoord)  # 4 x 4
        # Ensure ref_T_global is numpy array
        ref_T_global_4x4 = np.asarray(ref_T_global_4x4, dtype=np.float32)
        
        pix_T_cam = self._merge_intrinsics_4x4(self.optimal_intrinsic)
        
        # self.extrinsic is world-to-camera (4x4), so cam_T_global = self.extrinsic
        # Ensure it's a proper 4x4 matrix
        cam_T_global = np.asarray(self.extrinsic, dtype=np.float32).copy()
        if cam_T_global.shape != (4, 4):
            raise ValueError(f"extrinsic must be 4x4, got {cam_T_global.shape}")
        # Ensure last row is [0, 0, 0, 1] for proper homogeneous transformation
        cam_T_global[3, :] = [0, 0, 0, 1]
        
        # Compute transformations - work with 4x4 matrices, then extract 3x3 for 2D homography
        global_T_cam = np.linalg.inv(cam_T_global)
        ref_T_cam_4x4 = ref_T_global_4x4 @ global_T_cam
        # Ensure ref_T_cam is numpy array
        ref_T_cam_4x4 = np.asarray(ref_T_cam_4x4, dtype=np.float32)
        cam_T_ref_4x4 = np.linalg.inv(ref_T_cam_4x4)
        
        # Ensure cam_T_ref is numpy array (in case it's a Tensor)
        cam_T_ref_4x4 = np.asarray(cam_T_ref_4x4, dtype=np.float32)
        
        # Key: Extract 3x3 from 4x4 for 2D homography: rows [0,1,3] and columns [0,1,3]
        # This matches the original code: cam_T_ref[:3, [0, 1, 3]]
        pix_T_ref = pix_T_cam[:3, :3] @ cam_T_ref_4x4[:3, [0, 1, 3]]
        # Project mem points
        points_3d[:, -1] = 1.0
        
        # mem_homo = np.hstack([points_3d, np.ones((n_points, 1), dtype=np.float32)])  # Nx3
        img_coord = (pix_T_ref @ points_3d.T).T  # Nx3

        if undistorted_img is not None:
            img_ref = cv2.warpPerspective(undistorted_img, np.linalg.inv(pix_T_ref), (200, 160), flags=cv2.INTER_CUBIC)
        
        # Normalize
        img_coord = img_coord /  img_coord[:, -1].reshape(-1, 1)  # Nx3
        result = img_coord[:, :2]  # Nx2
        valid_mask = (result[:, 0] >= 0) & (result[:, 0] < IMAGE_WIDTH) \
                    & (result[:, 1] >= 0) & (result[:, 1] < IMAGE_HEIGHT)
        
        # Now apply distortion to get final pixel coordinates
        # Convert from optimal_intrinsic space back to original intrinsic space with distortion
        objp0 = np.array([
            (result[:, 0] - self.optimal_intrinsic[0, 2]) / self.optimal_intrinsic[0, 0],
            (result[:, 1] - self.optimal_intrinsic[1, 2]) / self.optimal_intrinsic[1, 1],
            np.zeros(n_points)
        ], dtype=np.float32)  # 3xN
        
        rvec = np.zeros(3, dtype=np.float32)
        tvec = np.zeros(3, dtype=np.float32)
        
        imgpoints, _ = cv2.projectPoints(objp0.T, rvec, tvec, self.intrinsic, self.distortion)
        result = imgpoints.reshape(-1, 2)
        
        # Mark invalid points as NaN
        result = result.astype(np.float32)
        result[~valid_mask] = np.nan
        
        return result
    
    # def project_3d_to_2d(
    #     self, 
    #     points_3d: np.ndarray, 
    #     apply_distortion: bool = True,
    #     from_mem: bool = False,
    #     bev_x: int = 1200,
    #     bev_y: int = 800,
    #     bev_bounds: Optional[List] = None,
    # ) -> np.ndarray:
    #     """
    #     Project 3D points to 2D image coordinates.
        
    #     If from_mem=True, applies full transformation chain:
    #         mem -> ref (ref_T_mem) -> world (worldgrid2worldcoord) -> camera (extrinsic) 
    #         -> pixel (intrinsic) -> distorted
        
    #     If from_mem=False, applies standard projection:
    #         world -> camera (extrinsic) -> pixel (intrinsic) -> distorted
        
    #     Args:
    #         points_3d: Nx3 array of points (mem coordinates if from_mem=True, else world coordinates)
    #         apply_distortion: Whether to apply lens distortion
    #         from_mem: If True, input is memory coordinates; apply ref_T_mem and worldgrid2worldcoord first
    #         bev_x: BEV grid X dimension (only used if from_mem=True)
    #         bev_y: BEV grid Y dimension (only used if from_mem=True)
    #         bev_bounds: BEV bounds (only used if from_mem=True)
            
    #     Returns:
    #         Nx2 array of 2D image coordinates, NaN for points behind camera.
    #     """
    #     if points_3d.shape[1] != 3:
    #         raise ValueError(f"points_3d must be Nx3, got {points_3d.shape}")
        
    #     # Ensure float32
    #     points_3d = np.asarray(points_3d, dtype=np.float32)
    #     n_points = points_3d.shape[0]
        
    #     if bev_bounds is None:
    #         bev_bounds = DEFAULT_BEV_BOUNDS
        
    #     device = 'cpu'
    #     B = 1
    #     X, Y, Z = bev_x, bev_y, 2
        
    #     # Create VoxelUtil
    #     scene_centroid = torch.tensor([0.0, 0.0, 0.0]).reshape([1, 3])
    #     vox_util = vox.VoxelUtil(Y, Z, X, scene_centroid=scene_centroid, bounds=bev_bounds)
        
    #     # Get transformation matrices
    #     ref_T_mem = vox_util.get_ref_T_mem(B, Y, Z, X, device=device)  # B x 4 x 4
    #     worldgrid2worldcoord = get_worldgrid2worldcoord_torch(device)  # 3 x 3 
        
    #     # Extract 3x3 submatrix for 2D homography (rows [0,1,3], cols [0,1,3])
    #     ref_T_mem_2d = ref_T_mem[0, [0, 1, 3]][:, [0, 1, 3]]  # 3x3
        
    #     # Combined 2D transformation: worldcoord_T_mem = worldgrid2worldcoord @ ref_T_mem_2d
    #     worldcoord_T_mem = torch.matmul(worldgrid2worldcoord, ref_T_mem_2d)  # 3x3
        
    #     # Convert mem points (x, y, z) to homogeneous 2D (x, y, 1) for ground plane
    #     mem_pts_2d = torch.tensor(
    #         np.hstack([points_3d[:, :2], np.ones((n_points, 1))]),
    #         dtype=torch.float32, device=device
    #     )  # N x 3
        
    #     # Transform mem -> world coordinates (2D on ground plane)
    #     world_pts_homo = torch.matmul(mem_pts_2d, worldcoord_T_mem.T)  # N x 3
    #     world_2d = world_pts_homo[:, :2] / world_pts_homo[:, 2:3]  # N x 2
        
    #     # Create 3D world points using transformed x,y and original z
    #     points_3d = np.zeros((n_points, 3), dtype=np.float32)
    #     points_3d[:, 0] = world_2d[:, 0].numpy()
    #     points_3d[:, 1] = world_2d[:, 1].numpy()
    #     points_3d[:, 2] = 0.0  # Ground plane (z from mem is typically 0)
                
    #     # Compute camera coordinates to know which points are in front of camera
    #     points_3d_homogeneous = np.hstack([points_3d, np.ones((n_points, 1), dtype=np.float32)])
    #     points_cam = (self.extrinsic @ points_3d_homogeneous.T).T
    #     valid_mask = points_cam[:, 2] > 0  # z > 0 in camera frame
        
    #     # Use OpenCV's projectPoints for consistency with calibration files
    #     R = self.extrinsic[:3, :3]
    #     t = self.extrinsic[:3, 3]
    #     rvec, _ = cv2.Rodrigues(R)
        
    #     # Apply distortion if requested
    #     dist_coeffs = self.distortion 
        
    #     image_points, _ = cv2.projectPoints(
    #         points_3d,
    #         rvec,
    #         t,
    #         self.intrinsic,
    #         dist_coeffs,
    #     )
    #     result = image_points.reshape(-1, 2)
        
    #     # Mark invalid points as NaN
    #     result = result.astype(np.float32)
    #     result[~valid_mask] = np.nan
        
    #     return result

    def project_mem_to_2d(
        self, 
        mem_points: np.ndarray,
        bev_x: int = 1200,
        bev_y: int = 800,
        bev_bounds: Optional[List] = None,
        apply_distortion: bool = True,
    ) -> np.ndarray:
        """
        Project BEV memory coordinates to 2D image coordinates.
        
        Full transformation chain using ref_T_mem and worldgrid2worldcoord:
            mem -> ref (via ref_T_mem) -> world (via worldgrid2worldcoord) -> world_3d (z=0)
            -> camera (via extrinsic) -> pixel (via intrinsic) -> distorted (via distortion)
        
        Args:
            mem_points: Nx2 array of memory (BEV pixel) coordinates
            bev_x: BEV grid X dimension
            bev_y: BEV grid Y dimension  
            bev_bounds: BEV bounds [XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX]
            apply_distortion: Whether to apply lens distortion
            
        Returns:
            Nx2 array of 2D image coordinates, NaN for invalid points.
        """
        n_points = mem_points.shape[0]
        
        # Convert 2D mem points to 3D (with z=0)
        points_3d = np.zeros((n_points, 3), dtype=np.float32)
        points_3d[:, 0] = mem_points[:, 0]
        points_3d[:, 1] = mem_points[:, 1]
        points_3d[:, 2] = 0.0  # Ground plane
        
        # Use project_3d_to_2d with from_mem=True to apply full transformation chain
        result = self.project_3d_to_2d(
            points_3d, 
            apply_distortion=apply_distortion,
            from_mem=True,
            bev_x=bev_x,
            bev_y=bev_y,
            bev_bounds=bev_bounds
        )
        
        return result
    
    def project_2d_to_mem(
        self,
        pixel_points: np.ndarray,
        bev_x: int = 1200,
        bev_y: int = 800,
        bev_bounds: Optional[List] = None,
        ground_z: float = 0.0,
        apply_undistortion: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Project 2D image pixel coordinates to BEV memory coordinates.
        
        Inverse transformation chain:
            distorted pixel -> undistorted pixel -> camera -> world (z=ground_z) 
            -> ref (via worldgrid2worldcoord^-1) -> mem (via ref_T_mem^-1)
        
        Args:
            pixel_points: Nx2 array of pixel coordinates in image
            bev_x: BEV grid X dimension
            bev_y: BEV grid Y dimension
            bev_bounds: BEV bounds [XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX]
            ground_z: Height of ground plane in world coordinates (default 0.0)
            apply_undistortion: Whether to apply undistortion (default True)
            
        Returns:
            Nx2 array of memory coordinates, or None if projection fails
        """
        from argustrack.utils.vox_utils import vox
        from argustrack.utils.constants import get_worldgrid2worldcoord_torch, DEFAULT_BEV_BOUNDS
        import torch
        
        if pixel_points.shape[1] != 2:
            raise ValueError(f"pixel_points must be Nx2, got {pixel_points.shape}")
        
        if bev_bounds is None:
            bev_bounds = DEFAULT_BEV_BOUNDS
        
        n_points = pixel_points.shape[0]
        pixel_points = np.asarray(pixel_points, dtype=np.float32)
        
        # Step 1: Undistort pixel coordinates if needed
        if apply_undistortion:
            # Undistort using optimal intrinsic
            pixel_points_reshaped = pixel_points.reshape(-1, 1, 2)
            undistorted = cv2.undistortPoints(
                pixel_points_reshaped,
                self.intrinsic,
                self.distortion,
                None,
                self.optimal_intrinsic
            )
            pixel_points_normalized = undistorted.reshape(-1, 2)
        else:
            # Convert to normalized coordinates using optimal intrinsic
            pixel_points_normalized = np.zeros_like(pixel_points)
            pixel_points_normalized[:, 0] = (pixel_points[:, 0] - self.optimal_intrinsic[0, 2]) / self.optimal_intrinsic[0, 0]
            pixel_points_normalized[:, 1] = (pixel_points[:, 1] - self.optimal_intrinsic[1, 2]) / self.optimal_intrinsic[1, 1]
        
        # Step 2: Transform directly from pixel to mem coordinates
        # Use same transformation chain as project_3d_to_2d but in reverse
        # In project_3d_to_2d: mem (x, y, 1) -> ref (via pix_T_ref) -> pixel
        # So here: pixel -> ref (via pix_T_ref^-1) -> mem
        
        device = 'cpu'
        worldgrid2worldcoord = get_worldgrid2worldcoord_torch(device)  # 4 x 4 (homogeneous)
        if isinstance(worldgrid2worldcoord, torch.Tensor):
            worldgrid2worldcoord = worldgrid2worldcoord.cpu().numpy()
        
        # worldgrid2worldcoord is 4x4, use it directly as 4x4
        ref_T_global_4x4 = np.linalg.inv(worldgrid2worldcoord)  # 4 x 4
        ref_T_global_4x4 = np.asarray(ref_T_global_4x4, dtype=np.float32)
        
        pix_T_cam = self._merge_intrinsics_4x4(self.optimal_intrinsic)
        
        # self.extrinsic is world-to-camera (4x4), so cam_T_global = self.extrinsic
        # Ensure it's a proper 4x4 matrix
        cam_T_global = np.asarray(self.extrinsic, dtype=np.float32).copy()
        if cam_T_global.shape != (4, 4):
            raise ValueError(f"extrinsic must be 4x4, got {cam_T_global.shape}")
        # Ensure last row is [0, 0, 0, 1] for proper homogeneous transformation
        cam_T_global[3, :] = [0, 0, 0, 1]
        
        global_T_cam = np.linalg.inv(cam_T_global)
        
        ref_T_cam_4x4 = ref_T_global_4x4 @ global_T_cam
        ref_T_cam_4x4 = np.asarray(ref_T_cam_4x4, dtype=np.float32)
        cam_T_ref_4x4 = np.linalg.inv(ref_T_cam_4x4)
        cam_T_ref_4x4 = np.asarray(cam_T_ref_4x4, dtype=np.float32)
        
        # Compute pix_T_ref (same as project_3d_to_2d)
        # pix_T_ref: 3x3 matrix that transforms mem (x, y, 1) to pixel
        # Extract 3x3 from 4x4: rows [0,1,3] and columns [0,1,3]
        pix_T_ref = pix_T_cam[:3, :3] @ cam_T_ref_4x4[:3, [0, 1, 3]]  # 3x3
        
        # Ensure pix_T_ref is numpy array
        pix_T_ref = np.asarray(pix_T_ref, dtype=np.float32)
        if pix_T_ref.shape != (3, 3):
            raise ValueError(f"pix_T_ref has unexpected shape: {pix_T_ref.shape}, expected (3, 3)")
        
        # Inverse: mem_T_pix = (pix_T_ref)^-1
        # This transforms pixel back to mem coordinates
        mem_T_pix = np.linalg.inv(pix_T_ref)  # 3x3
        
        # Transform pixel coordinates to mem coordinates
        # Convert normalized pixel points to homogeneous coordinates (x, y, 1)
        pixel_pts_homo = np.ones((n_points, 3), dtype=np.float32)
        pixel_pts_homo[:, 0] = pixel_points_normalized[:, 0]
        pixel_pts_homo[:, 1] = pixel_points_normalized[:, 1]
        
        # Apply transformation: mem = mem_T_pix @ pixel
        mem_pts_homo = (mem_T_pix @ pixel_pts_homo.T).T  # N x 3
        mem_points_result = mem_pts_homo[:, :2] / (mem_pts_homo[:, 2:3] + 1e-8)  # N x 2
        
        # Check validity: points should be finite
        valid_mask = np.isfinite(mem_points_result).all(axis=1)
        mem_points_result[~valid_mask] = np.nan
        
        # Clamp mem coordinates to BEV bounds to ensure they're within valid range
        # This prevents points from being displayed incorrectly on BEV canvas
        if bev_bounds is not None and len(bev_bounds) >= 4:
            x_min, x_max = bev_bounds[0], bev_bounds[1]
            y_min, y_max = bev_bounds[2], bev_bounds[3]
            # Only clamp valid points
            valid_indices = valid_mask
            if np.any(valid_indices):
                mem_points_result[valid_indices, 0] = np.clip(mem_points_result[valid_indices, 0], x_min, x_max)
                mem_points_result[valid_indices, 1] = np.clip(mem_points_result[valid_indices, 1], y_min, y_max)
        
        return mem_points_result

    
    def project_3d_box_to_2d(
        self,
        center_3d: np.ndarray,
        size_3d: np.ndarray,
        rotation: float = 0.0,
    ) -> Optional[np.ndarray]:
        """
        Project a 3D bounding box to 2D image coordinates.

        This follows the same convention as `anylabeling`'s Box3D:
        - center_3d: [x, y, z] where x,y are ground-plane coordinates, z is bottom height.
        - size_3d: [width, height, depth] where:
            * width  (w) is lateral size on ground (y-axis),
            * depth  (d) is longitudinal size on ground (x-axis),
            * height (h) is vertical size (z-axis).
        - rotation: rotation around the vertical (z) axis in radians.

        Returns:
            8x2 array of 2D corner coordinates, or None if box is not visible.
        """
        w, h, d = size_3d  # width (y), height (z), depth/length (x)
        cx, cy, cz = center_3d

        # 1) Build 2D rectangle on ground plane (x,y) with length=d, width=w
        half_w = w / 2.0
        half_l = d / 2.0

        # Local 2D corners before rotation (same order as anylabeling Box3D.get_corners_2d)
        corners_local = np.array(
            [
                [-half_l, -half_w],
                [half_l, -half_w],
                [half_l, half_w],
                [-half_l, half_w],
            ],
            dtype=np.float32,
        )

        # Apply rotation around z-axis (in ground plane)
        if rotation != 0.0:
            cos_r = np.cos(rotation)
            sin_r = np.sin(rotation)
            rot_2d = np.array([[cos_r, -sin_r], [sin_r, cos_r]], dtype=np.float32)
            corners_rot = (rot_2d @ corners_local.T).T
        else:
            corners_rot = corners_local

        # Translate to world (x,y), bottom face at z = cz
        bottom = np.zeros((4, 3), dtype=np.float32)
        bottom[:, 0] = corners_rot[:, 0] + cx  # x
        bottom[:, 1] = corners_rot[:, 1] + cy  # y
        bottom[:, 2] = cz                      # z

        # Top face at z = cz + h
        top = bottom.copy()
        top[:, 2] = cz + h

        corners_3d = np.vstack([bottom, top])  # 8x3

        # Project to 2D
        corners_2d = self.project_3d_to_2d(
            corners_3d,
            bev_x=DEFAULT_BEV_X,
            bev_y=DEFAULT_BEV_Y,
            bev_bounds=DEFAULT_BEV_BOUNDS,
        )

        # If all projected points are invalid, box is not visible
        if np.all(np.isnan(corners_2d)):
            return None

        return corners_2d
    
    def get_2d_bbox_from_3d_box(self, center_3d: np.ndarray, size_3d: np.ndarray,
                                rotation: float = 0.0) -> Optional[np.ndarray]:
        """
        Get 2D bounding box from 3D box projection
        
        Args:
            center_3d: [x, y, z] center of box in world coordinates
            size_3d: [width, height, depth] size of box
            rotation: rotation angle around z-axis (in radians)
            
        Returns:
            [x_min, y_min, x_max, y_max] or None if box is not visible
        """
        corners_2d = self.project_3d_box_to_2d(center_3d, size_3d, rotation)
        if corners_2d is None:
            return None
        
        # Filter out NaN values
        valid_corners = corners_2d[~np.isnan(corners_2d).any(axis=1)]
        if len(valid_corners) == 0:
            return None
        
        x_min = np.min(valid_corners[:, 0])
        y_min = np.min(valid_corners[:, 1])
        x_max = np.max(valid_corners[:, 0])
        y_max = np.max(valid_corners[:, 1])
        
        return np.array([x_min, y_min, x_max, y_max])


def generate_bev_from_cameras(
    camera_calibrations: Dict[str, CameraCalibration],
    camera_data: List[Dict[str, Any]],
    bev_width: float,
    bev_height: float,
    resolution: float = 0.2,
    bev_x: int = 1200,
    bev_y: int = 800,
    bev_bounds: Optional[List] = None,
) -> Optional[np.ndarray]:
    """
    Generate a simple BEV (top-down) image from multiple cameras using cv2.warpPerspective.
    
    Uses perspective transformation to warp undistorted camera images directly to BEV space.

    Args:
        camera_calibrations: mapping camera_id -> CameraCalibration.
        camera_data: list of dicts with at least keys: 'camera_id', 'image_path'.
        bev_width: BEV width (used for display bounds).
        bev_height: BEV height (used for display bounds).
        resolution: meters per pixel (smaller -> higher resolution).
        bev_x: BEV grid X dimension
        bev_y: BEV grid Y dimension
        bev_bounds: BEV bounds [XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX]

    Returns:
        HxWx3 uint8 RGB image, or None if generation fails.
    """
    import cv2  # type: ignore[import]
    from argustrack.utils.constants import DEFAULT_BEV_BOUNDS

    if not camera_calibrations or not camera_data:
        return None
    
    if bev_bounds is None:
        bev_bounds = DEFAULT_BEV_BOUNDS

    # Use BEV grid dimensions directly
    n_x = bev_x
    n_y = bev_y
    
    # Initialize accumulation arrays for blending
    accum = np.zeros((n_y, n_x, 3), dtype=np.float32)
    counts = np.zeros((n_y, n_x), dtype=np.float32)

    device = 'cpu'
    B = 1
    X, Y, Z = bev_x, bev_y, 2
    worldgrid2worldcoord = get_worldgrid2worldcoord_torch(device)  # 3 x 3 
    # Convert Tensor to numpy array to avoid type mismatch
    if isinstance(worldgrid2worldcoord, torch.Tensor):
        worldgrid2worldcoord = worldgrid2worldcoord.cpu().numpy()
    ref_T_global = np.linalg.inv(worldgrid2worldcoord)  # 3 x 3
    # Ensure ref_T_global is numpy array
    ref_T_global = np.asarray(ref_T_global, dtype=np.float32)

    for cam in camera_data:
        camera_id = cam.get("camera_id")
        image_path = cam.get("image_path")
        if not camera_id or not image_path:
            continue

        calib = camera_calibrations.get(camera_id)
        if calib is None:
            continue

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning(f"Failed to read image for BEV: {image_path}")
            continue

        h, w = img.shape[:2]
        
        try:
            # Get undistorted image
            undistorted_img = calib._get_undistorted_img(img)
            
            # Compute transformation matrices (same as in project_3d_to_2d)
            pix_T_cam = calib._merge_intrinsics_4x4(calib.optimal_intrinsic)
            
            # calib.extrinsic is world-to-camera (4x4), so cam_T_global = calib.extrinsic
            # Ensure it's a proper 4x4 matrix
            cam_T_global = np.asarray(calib.extrinsic, dtype=np.float32).copy()
            if cam_T_global.shape != (4, 4):
                raise ValueError(f"extrinsic must be 4x4, got {cam_T_global.shape}")
            # Ensure last row is [0, 0, 0, 1] for proper homogeneous transformation
            cam_T_global[3, :] = [0, 0, 0, 1]
            
            global_T_cam = np.linalg.inv(cam_T_global)
            ref_T_cam = ref_T_global @ global_T_cam
            # Ensure ref_T_cam is numpy array
            ref_T_cam = np.asarray(ref_T_cam, dtype=np.float32)
            cam_T_ref = np.linalg.inv(ref_T_cam)
            # Ensure cam_T_ref is numpy array
            cam_T_ref = np.asarray(cam_T_ref, dtype=np.float32)
            
            # 3x3 projection matrix from mem/ref to pixel coordinates
            pix_T_ref = pix_T_cam[:3, :3] @ cam_T_ref[:3, [0, 1, 3]]  # 3x3
            
            # Inverse transformation: from pixel to mem/ref coordinates
            ref_T_pix = np.linalg.inv(pix_T_ref)
            
            # Warp undistorted image to BEV space
            # Output size is (bev_x, bev_y) -> (width, height) in OpenCV convention
            warped_bgr = cv2.warpPerspective(
                undistorted_img,
                ref_T_pix,
                (n_x, n_y),
                flags=cv2.INTER_LINEAR,
            )
            
            # Convert BGR to RGB
            warped_rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            
            # Create mask for valid pixels (non-black pixels)
            mask = (warped_rgb.sum(axis=2) > 0).astype(np.float32)
            
            # Accumulate weighted by mask
            accum += warped_rgb * mask[:, :, np.newaxis]
            counts += mask
            
            logger.info(f"Successfully warped camera {camera_id} to BEV")
            
        except Exception as e:
            logger.warning(f"Failed to warp camera {camera_id} to BEV: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Check if we have any valid pixels
    valid_mask = counts > 0
    if not np.any(valid_mask):
        logger.warning("No valid BEV pixels generated from cameras")
        return None

    # Average the accumulated colors
    bev = np.zeros((n_y, n_x, 3), dtype=np.uint8)
    bev[valid_mask] = (accum[valid_mask] / counts[valid_mask, np.newaxis]).astype(np.uint8)

    # Post-process for better visibility: enhance contrast while preserving original colors
    try:
        # Convert to LAB color space for better contrast enhancement
        # LAB preserves color information better than RGB when enhancing brightness/contrast
        bev_lab = cv2.cvtColor(bev, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(bev_lab)
        
        # Apply CLAHE to L channel only (lightness) to enhance contrast while preserving color channels
        # Use moderate clipLimit to avoid over-enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)
        
        # Slightly increase brightness to make it more visible
        # Scale L channel moderately to preserve natural look
        l_enhanced = np.clip(l_enhanced.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
        
        # Merge back with original color channels (a and b) to preserve original colors
        bev_lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        bev_rgb = cv2.cvtColor(bev_lab_enhanced, cv2.COLOR_LAB2RGB)
        
        # Apply slight saturation boost to make colors more vibrant (optional)
        # Convert to HSV, slightly boost saturation, convert back
        bev_hsv = cv2.cvtColor(bev_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        h, s, v = cv2.split(bev_hsv)
        # Boost saturation slightly (1.15x) to make colors pop while keeping original hue
        s_enhanced = np.clip(s * 1.15, 0, 255).astype(np.uint8)
        bev_hsv_enhanced = cv2.merge([h.astype(np.uint8), s_enhanced, v.astype(np.uint8)])
        bev_rgb = cv2.cvtColor(bev_hsv_enhanced, cv2.COLOR_HSV2RGB)
        
        return bev_rgb
    except Exception as e:
        logger.warning(f"Failed to enhance BEV image contrast, using raw mosaic: {e}")
        return bev
        