"""
Metadata management for BEV annotations.

This module handles saving and loading metadata files (constants.json)
that store BEV grid configuration and default box dimensions.
"""

import json
import os
import os.path as osp
from typing import Dict, Any, Optional, Tuple
from loguru import logger

from .constants import (
    DEFAULT_BEV_X,
    DEFAULT_BEV_Y,
    DEFAULT_BEV_Z,
    DEFAULT_BEV_BOUNDS,
    DEFAULT_BOX_WIDTH,
    DEFAULT_BOX_HEIGHT,
    DEFAULT_BOX_DEPTH,
    WORLD_GRID_THETA,
    WORLD_GRID_SCALE,
    WORLD_GRID_TX,
    WORLD_GRID_TY,
)


def save_metadata(
    root_dir: str,
    bev_x: Optional[int] = None,
    bev_y: Optional[int] = None,
    bev_z: Optional[int] = None,
    bev_bounds: Optional[list] = None,
    box_width: Optional[float] = None,
    box_height: Optional[float] = None,
    box_depth: Optional[float] = None,
) -> bool:
    """
    Save metadata (constants) to constants.json at the root directory level.
    
    Args:
        root_dir: Root directory (same level as Image_subsets)
        bev_x: BEV grid X dimension (pixels)
        bev_y: BEV grid Y dimension (pixels)
        bev_z: BEV grid Z dimension (height levels)
        bev_bounds: BEV bounds [XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX]
        box_width: Default box width in world grid units
        box_height: Default box height in world grid units
        box_depth: Default box depth in world grid units
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        metadata = {
            "version": "1.0",
            "bev_grid": {
                "x": bev_x if bev_x is not None else DEFAULT_BEV_X,
                "y": bev_y if bev_y is not None else DEFAULT_BEV_Y,
                "z": bev_z if bev_z is not None else DEFAULT_BEV_Z,
            },
            "bev_bounds": bev_bounds if bev_bounds is not None else DEFAULT_BEV_BOUNDS,
            "default_box_size": {
                "width": box_width if box_width is not None else DEFAULT_BOX_WIDTH,
                "height": box_height if box_height is not None else DEFAULT_BOX_HEIGHT,
                "depth": box_depth if box_depth is not None else DEFAULT_BOX_DEPTH,
            },
            "world_grid_transform": {
                "theta": WORLD_GRID_THETA,
                "scale": WORLD_GRID_SCALE,
                "tx": WORLD_GRID_TX,
                "ty": WORLD_GRID_TY,
            },
        }
        
        metadata_path = osp.join(root_dir, "constants.json")
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_metadata(root_dir: str) -> Dict[str, Any]:
    """
    Load metadata (constants) from constants.json.
    
    Args:
        root_dir: Root directory (same level as Image_subsets)
    
    Returns:
        Dict containing metadata with keys:
            - bev_grid: {x, y, z}
            - bev_bounds: [XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX]
            - default_box_size: {width, height, depth}
            - world_grid_transform: {theta, scale, tx, ty}
        
        If file doesn't exist or fails to load, returns defaults from constants.py
    """
    metadata_path = osp.join(root_dir, "constants.json")
    
    # Default values
    default_metadata = {
        "version": "1.0",
        "bev_grid": {
            "x": DEFAULT_BEV_X,
            "y": DEFAULT_BEV_Y,
            "z": DEFAULT_BEV_Z,
        },
        "bev_bounds": DEFAULT_BEV_BOUNDS,
        "default_box_size": {
            "width": DEFAULT_BOX_WIDTH,
            "height": DEFAULT_BOX_HEIGHT,
            "depth": DEFAULT_BOX_DEPTH,
        },
        "world_grid_transform": {
            "theta": WORLD_GRID_THETA,
            "scale": WORLD_GRID_SCALE,
            "tx": WORLD_GRID_TX,
            "ty": WORLD_GRID_TY,
        },
    }
    
    if not osp.exists(metadata_path):
        logger.info(f"No metadata file found at {metadata_path}, using defaults")
        return default_metadata
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded metadata from {metadata_path}")
        
        # Merge with defaults (in case some keys are missing)
        for key in default_metadata:
            if key not in metadata:
                metadata[key] = default_metadata[key]
        
        return metadata
        
    except Exception as e:
        logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
        import traceback
        traceback.print_exc()
        logger.info("Using default metadata values")
        return default_metadata


def get_bev_grid_from_metadata(metadata: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Extract BEV grid dimensions from metadata.
    
    Args:
        metadata: Metadata dictionary
    
    Returns:
        Tuple of (x, y, z) grid dimensions
    """
    bev_grid = metadata.get("bev_grid", {})
    return (
        bev_grid.get("x", DEFAULT_BEV_X),
        bev_grid.get("y", DEFAULT_BEV_Y),
        bev_grid.get("z", DEFAULT_BEV_Z),
    )


def get_box_size_from_metadata(metadata: Dict[str, Any]) -> Tuple[float, float, float]:
    """
    Extract default box size from metadata.
    
    Args:
        metadata: Metadata dictionary
    
    Returns:
        Tuple of (width, height, depth) in world grid units
    """
    box_size = metadata.get("default_box_size", {})
    return (
        box_size.get("width", DEFAULT_BOX_WIDTH),
        box_size.get("height", DEFAULT_BOX_HEIGHT),
        box_size.get("depth", DEFAULT_BOX_DEPTH),
    )


if __name__ == "__main__":
    # Test metadata saving and loading
    import tempfile
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Testing metadata in {tmpdir}")
        
        # Test save
        success = save_metadata(
            tmpdir,
            bev_x=800,
            bev_y=600,
            bev_z=3,
            box_width=60.0,
            box_height=70.0,
            box_depth=80.0,
        )
        assert success, "Failed to save metadata"
        print("✓ Save metadata successful")
        
        # Test load
        metadata = load_metadata(tmpdir)
        assert metadata["bev_grid"]["x"] == 800
        assert metadata["bev_grid"]["y"] == 600
        assert metadata["bev_grid"]["z"] == 3
        assert metadata["default_box_size"]["width"] == 60.0
        assert metadata["default_box_size"]["height"] == 70.0
        assert metadata["default_box_size"]["depth"] == 80.0
        print("✓ Load metadata successful")
        
        # Test helper functions
        bev_x, bev_y, bev_z = get_bev_grid_from_metadata(metadata)
        assert (bev_x, bev_y, bev_z) == (800, 600, 3)
        print("✓ Get BEV grid successful")
        
        w, h, d = get_box_size_from_metadata(metadata)
        assert (w, h, d) == (60.0, 70.0, 80.0)
        print("✓ Get box size successful")
        
        # Test load defaults
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir2:
            metadata_default = load_metadata(tmpdir2)
            assert metadata_default["bev_grid"]["x"] == DEFAULT_BEV_X
            assert metadata_default["default_box_size"]["width"] == DEFAULT_BOX_WIDTH
            print("✓ Load defaults successful")
        
        print("\n✅ All tests passed!")











