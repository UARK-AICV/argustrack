#!/usr/bin/env python3
"""
Utility functions for working with action zones.
Determines which action zone a point falls into.
"""

import json
import numpy as np
from typing import List, Tuple, Optional
import cv2
from pathlib import Path


class ActionZoneMapper:
    """Maps BEV coordinates to action zones"""
    
    def __init__(self, zone_file: str, image_width: Optional[int] = None, 
                 image_height: Optional[int] = None):
        """
        Initialize action zone mapper
        
        Args:
            zone_file: Path to action zones JSON file
            image_width: Width of BEV image (for normalized coordinates)
            image_height: Height of BEV image (for normalized coordinates)
        """
        self.zone_file = zone_file
        self.zones = []
        self.zones_normalized = []
        self.default_action = "walking"
        self.image_width = image_width
        self.image_height = image_height
        
        self._load_zones()
    
    def _load_zones(self) -> None:
        """Load action zones from JSON file"""
        try:
            if not Path(self.zone_file).exists():
                print(f"Warning: Zone file not found: {self.zone_file}")
                return
            
            with open(self.zone_file, 'r') as f:
                data = json.load(f)
                
                # Get image size from file if not provided
                if self.image_width is None or self.image_height is None:
                    image_size = data.get("image_size", {})
                    self.image_width = image_size.get("width", 1200)
                    self.image_height = image_size.get("height", 800)
                
                zones = data.get("zones", [])
                
                # Store both pixel and normalized versions
                for zone in zones:
                    # Try normalized coordinates first
                    points_normalized = zone.get("points_normalized", [])
                    if points_normalized:
                        # Convert normalized to pixel coordinates
                        points_pixels = [
                            (int(nx * self.image_width), int(ny * self.image_height))
                            for nx, ny in points_normalized
                        ]
                    else:
                        # Use pixel coordinates directly
                        points_pixels = zone.get("points", [])
                        # Convert to normalized
                        points_normalized = [
                            (x / self.image_width, y / self.image_height)
                            for x, y in points_pixels
                        ]
                    
                    zone_data = {
                        "action": zone.get("action"),
                        "points": points_pixels,
                        "points_normalized": points_normalized
                    }
                    self.zones.append(zone_data)
            
            print(f"Loaded {len(self.zones)} action zones")
            print(f"Image size: {self.image_width}x{self.image_height}")
        except Exception as e:
            print(f"Error loading zones: {e}")
    
    def get_action_for_point(self, x: float, y: float, normalized: bool = False) -> str:
        """
        Get action for a point based on which zone it falls into.
        
        Args:
            x: X coordinate in BEV space
            y: Y coordinate in BEV space
            normalized: If True, x,y are in 0-1 range; if False, pixel coordinates
        
        Returns:
            Action name (walking, eating, sitting, standing)
        """
        if normalized:
            # Convert normalized to pixel coordinates
            point = (int(x * self.image_width), int(y * self.image_height))
        else:
            point = (int(x), int(y))
        
        # Check each zone (later zones take priority)
        for zone in self.zones:
            points = zone.get("points", [])
            if len(points) < 3:
                continue
            
            # Create polygon contour
            contour = np.array(points, dtype=np.int32)
            
            # Check if point is inside polygon
            result = cv2.pointPolygonTest(contour, point, False)
            if result >= 0:  # Inside or on edge
                return zone.get("action", self.default_action)
        
        # Not in any zone, return default
        return self.default_action
    
    def get_all_zones(self) -> List[dict]:
        """Get all loaded zones"""
        return self.zones
    
    def visualize_zones(self, image: np.ndarray) -> np.ndarray:
        """
        Draw all zones on an image for visualization
        
        Args:
            image: Input image (will be copied)
        
        Returns:
            Image with zones drawn
        """
        output = image.copy()
        
        ACTION_COLORS = {
            "walking": (0, 255, 0),      # Green
            "eating": (0, 165, 255),     # Orange
            "sitting": (255, 0, 0),      # Blue
            "standing": (0, 255, 255),   # Yellow
        }
        
        for zone in self.zones:
            points = zone.get("points", [])
            action = zone.get("action", "walking")
            
            if len(points) < 3:
                continue
            
            # Draw filled polygon with transparency
            pts = np.array(points, dtype=np.int32)
            color = ACTION_COLORS.get(action, (128, 128, 128))
            
            overlay = output.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.3, output, 0.7, 0, output)
            
            # Draw border
            cv2.polylines(output, [pts], True, color, 2)
            
            # Draw label
            cx = int(np.mean([p[0] for p in points]))
            cy = int(np.mean([p[1] for p in points]))
            cv2.putText(output, action, (cx, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return output


def test_zone_mapper():
    """Test the action zone mapper"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python action_zone_utils.py <zone_file> [test_x] [test_y]")
        print("Example: python action_zone_utils.py action_zones.json 100 200")
        return
    
    zone_file = sys.argv[1]
    mapper = ActionZoneMapper(zone_file)
    
    if len(sys.argv) >= 4:
        # Test specific point
        x = float(sys.argv[2])
        y = float(sys.argv[3])
        action = mapper.get_action_for_point(x, y)
        print(f"\nPoint ({x}, {y}) is in zone: {action}")
    
    # Print all zones
    print("\nLoaded zones:")
    for i, zone in enumerate(mapper.get_all_zones(), 1):
        action = zone.get("action")
        num_points = len(zone.get("points", []))
        print(f"  {i}. {action} zone ({num_points} vertices)")


if __name__ == "__main__":
    test_zone_mapper()
