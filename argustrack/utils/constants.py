"""
Constants for BEV transformation and projection.

These values are used for coordinate transformations between:
- Memory coordinates (BEV grid pixels)
- Reference coordinates (voxel world)
- World grid coordinates
- World coordinates (meters)
"""

import numpy as np
import torch
from typing import Tuple, List

# ============================================================================
# Default 3D Box Dimensions
# ============================================================================

# Default box size (width, height, depth) in world grid units
DEFAULT_BOX_WIDTH = 40.0  # Width of bounding box
DEFAULT_BOX_HEIGHT = 40.0  # Height of bounding box  
DEFAULT_BOX_DEPTH = 40.0  # Depth of bounding box
DEFAULT_BOX_SIZE = 40.0  # Legacy: kept for backward compatibility

IMAGE_WIDTH = 960
IMAGE_HEIGHT = 540

# ============================================================================
# World Grid to World Coordinate Transformation
# ============================================================================

# Rotation angle for world grid alignment
WORLD_GRID_THETA = -2  # degrees
WORLD_GRID_THETA_RAD = WORLD_GRID_THETA / 180.0 * np.pi

# Scale factor for grid to world conversion
WORLD_GRID_SCALE = 2.5 / 4

# Translation offset for grid to world conversion
WORLD_GRID_TX = -76
WORLD_GRID_TY = -112

# World grid to world coordinate transformation matrix (3x3 homogeneous)
# Converts world grid coordinates to world coordinates (meters)
WORLDGRID2WORLDCOORD_MAT = np.array([
    [WORLD_GRID_SCALE * np.cos(WORLD_GRID_THETA_RAD), WORLD_GRID_SCALE * -np.sin(WORLD_GRID_THETA_RAD), 0, WORLD_GRID_TX],
    [WORLD_GRID_SCALE * np.sin(WORLD_GRID_THETA_RAD), WORLD_GRID_SCALE *  np.cos(WORLD_GRID_THETA_RAD), 0,WORLD_GRID_TY],
    [0.,                     0.,                     1. ,   0.],
    [0.,                     0.,                     0. ,   1.],
], dtype=np.float32)

# Inverse: World coordinate to world grid
WORLDCOORD2WORLDGRID_MAT = np.linalg.inv(WORLDGRID2WORLDCOORD_MAT).astype(np.float32)


# ============================================================================
# BEV Grid Parameters
# ============================================================================

# Default BEV grid dimensions
DEFAULT_BEV_X = int(200) * 4  # Grid X dimension (pixels)
DEFAULT_BEV_Y = int(160 ) * 4 # Grid Y dimension (pixels)
DEFAULT_BEV_Z = 2     # Grid Z dimension (height levels)

# Default BEV bounds in world units [XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX]
DEFAULT_BEV_BOUNDS = [0, int(200) * 4, 0, int(160) * 4, 0, 2]


# ============================================================================
# Calibration Scale Factors
# ============================================================================

# Default intrinsic scale factor (for K and optimalK)
DEFAULT_INTRINSIC_SCALE = 0.25  # 1/4 scale

# Default translation scale factor (for extrinsic T)
DEFAULT_TRANSLATION_SCALE = 100.0


# ============================================================================
# Helper Functions
# ============================================================================

def get_worldgrid2worldcoord_torch(device: str = 'cpu') -> torch.Tensor:
    """Get world grid to world coordinate matrix as torch tensor."""
    return torch.tensor(WORLDGRID2WORLDCOORD_MAT, dtype=torch.float32, device=device)


def get_worldcoord2worldgrid_torch(device: str = 'cpu') -> torch.Tensor:
    """Get world coordinate to world grid matrix as torch tensor."""
    return torch.tensor(WORLDCOORD2WORLDGRID_MAT, dtype=torch.float32, device=device)


def get_default_bounds() -> List[int]:
    """Get default BEV bounds."""
    return DEFAULT_BEV_BOUNDS.copy()


def get_default_grid_dims() -> Tuple[int, int, int]:
    """Get default BEV grid dimensions (X, Y, Z)."""
    return DEFAULT_BEV_X, DEFAULT_BEV_Y, DEFAULT_BEV_Z

