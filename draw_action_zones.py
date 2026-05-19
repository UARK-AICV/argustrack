#!/usr/bin/env python3
"""
Simple script to draw action zone polygons on BEV overlay image.
Maps each polygon to an action: walking, eating, sitting, standing

Usage:
    python draw_action_zones.py

Instructions:
    - Left click: Add polygon vertex
    - Right click: Complete current polygon and assign action
    - 'u': Undo last vertex
    - 'c': Clear current polygon
    - 'd': Delete last completed polygon
    - 's': Save to JSON
    - 'q' or ESC: Quit
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Action types
ACTIONS = ["walking", "eating", "sitting", "standing", "drinking"]
ACTION_COLORS = {
    "walking": (0, 255, 0),      # Green
    "eating": (0, 165, 255),     # Orange
    "sitting": (255, 0, 0),      # Blue
    "standing": (0, 255, 255),   # Yellow
    "drinking": (255, 255, 0),   # Purple
}


class PolygonDrawer:
    """Interactive polygon drawing tool for action zones"""
    
    def __init__(self, image_path: str, output_json: str = "action_zones.json", 
                 display_scale: float = 1.5):
        """
        Initialize polygon drawer
        
        Args:
            image_path: Path to BEV overlay image
            output_json: Path to output JSON file
            display_scale: Scale factor for display (default 1.5 = 150% larger)
        """
        self.image_path = image_path
        self.output_json = output_json
        self.display_scale = display_scale
        
        # Load image
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Original dimensions
        self.height, self.width = self.original_image.shape[:2]
        
        # Scaled dimensions for display
        self.display_width = int(self.width * display_scale)
        self.display_height = int(self.height * display_scale)
        
        # Resize for display
        self.original_image_scaled = cv2.resize(
            self.original_image, 
            (self.display_width, self.display_height),
            interpolation=cv2.INTER_LINEAR
        )
        self.display_image = self.original_image_scaled.copy()
        
        logger.info(f"Original size: {self.width}x{self.height}")
        logger.info(f"Display size: {self.display_width}x{self.display_height} (scale={display_scale})")
        
        # Drawing state
        self.current_polygon: List[Tuple[int, int]] = []
        self.completed_polygons: List[dict] = []
        self.current_action_idx: int = 0
        self.auto_cycle_actions: bool = False  # Don't auto-cycle by default
        
        # Window name
        self.window_name = "Action Zone Drawer"
        
        # Load existing zones if available
        self._load_existing_zones()
    
    def _display_to_normalized(self, x: int, y: int) -> Tuple[float, float]:
        """Convert display coordinates to normalized coordinates (0-1 range)"""
        # First convert display to original image coordinates
        orig_x = x / self.display_scale
        orig_y = y / self.display_scale
        
        # Then normalize
        norm_x = orig_x / self.width
        norm_y = orig_y / self.height
        
        return (norm_x, norm_y)
    
    def _normalized_to_display(self, norm_x: float, norm_y: float) -> Tuple[int, int]:
        """Convert normalized coordinates (0-1 range) to display coordinates"""
        # First convert to original image coordinates
        orig_x = norm_x * self.width
        orig_y = norm_y * self.height
        
        # Then scale to display
        disp_x = int(orig_x * self.display_scale)
        disp_y = int(orig_y * self.display_scale)
        
        return (disp_x, disp_y)
    
    def _load_existing_zones(self) -> None:
        """Load existing action zones from JSON if file exists"""
        try:
            if Path(self.output_json).exists():
                with open(self.output_json, 'r') as f:
                    data = json.load(f)
                    zones = data.get("zones", [])
                    
                    # Convert normalized coordinates to display coordinates
                    for zone in zones:
                        points_normalized = zone.get("points_normalized", [])
                        if points_normalized:
                            # Use normalized coordinates
                            display_points = [
                                self._normalized_to_display(nx, ny) 
                                for nx, ny in points_normalized
                            ]
                            zone["points"] = display_points
                        else:
                            # Legacy: pixel coordinates, convert to display scale
                            pixel_points = zone.get("points", [])
                            display_points = [
                                (int(x * self.display_scale), int(y * self.display_scale))
                                for x, y in pixel_points
                            ]
                            zone["points"] = display_points
                    
                    self.completed_polygons = zones
                logger.info(f"Loaded {len(self.completed_polygons)} existing zones")
        except Exception as e:
            logger.warning(f"Could not load existing zones: {e}")
    
    def _save_zones(self) -> None:
        """Save action zones to JSON with normalized coordinates"""
        try:
            # Convert display coordinates to normalized coordinates
            zones_to_save = []
            for zone in self.completed_polygons:
                display_points = zone.get("points", [])
                
                # Convert to normalized coordinates
                points_normalized = [
                    self._display_to_normalized(x, y) 
                    for x, y in display_points
                ]
                
                # Also save pixel coordinates (original image size)
                points_pixels = [
                    (int(nx * self.width), int(ny * self.height))
                    for nx, ny in points_normalized
                ]
                
                zone_data = {
                    "action": zone.get("action"),
                    "points_normalized": points_normalized,  # Primary: normalized (0-1)
                    "points": points_pixels  # Secondary: pixel coordinates
                }
                zones_to_save.append(zone_data)
            
            data = {
                "image_path": self.image_path,
                "image_size": {
                    "width": self.width,
                    "height": self.height
                },
                "zones": zones_to_save
            }
            
            with open(self.output_json, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(zones_to_save)} zones to {self.output_json}")
            print(f"\n✓ Saved to {self.output_json}")
            print(f"  - Normalized coordinates (0-1 range)")
            print(f"  - Original size: {self.width}x{self.height}")
        except Exception as e:
            logger.error(f"Failed to save zones: {e}")
            print(f"\n✗ Error saving: {e}")
    
    def _draw_polygon(self, image: np.ndarray, points: List[Tuple[int, int]], 
                     action: str, is_current: bool = False) -> None:
        """
        Draw polygon on image
        
        Args:
            image: Image to draw on
            points: List of (x, y) points
            action: Action name
            is_current: If True, draw as current (incomplete) polygon
        """
        if len(points) < 2:
            return
        
        color = ACTION_COLORS.get(action, (128, 128, 128))
        
        # Draw lines
        pts = np.array(points, dtype=np.int32)
        if is_current:
            # Draw lines for incomplete polygon
            cv2.polylines(image, [pts], False, color, 2)
            # Draw vertices
            for pt in points:
                cv2.circle(image, pt, 4, color, -1)
        else:
            # Draw filled polygon with transparency
            overlay = image.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
            
            # Draw border
            cv2.polylines(image, [pts], True, color, 2)
            
            # Draw label
            if len(points) > 0:
                cx = int(np.mean([p[0] for p in points]))
                cy = int(np.mean([p[1] for p in points]))
                cv2.putText(image, action, (cx, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _redraw(self) -> None:
        """Redraw the display image"""
        self.display_image = self.original_image_scaled.copy()
        
        # Draw completed polygons
        for zone in self.completed_polygons:
            self._draw_polygon(self.display_image, zone["points"], zone["action"])
        
        # Draw current polygon being drawn
        if len(self.current_polygon) > 0:
            current_action = ACTIONS[self.current_action_idx]
            self._draw_polygon(self.display_image, self.current_polygon, 
                             current_action, is_current=True)
        
        # Draw instructions
        self._draw_instructions()
        
        cv2.imshow(self.window_name, self.display_image)
    
    def _draw_instructions(self) -> None:
        """Draw instructions on image"""
        auto_cycle_status = "ON" if self.auto_cycle_actions else "OFF"
        instructions = [
            f"Current Action: {ACTIONS[self.current_action_idx]} | Auto-cycle: {auto_cycle_status}",
            f"Display: {self.display_width}x{self.display_height} (scale={self.display_scale:.1f}x)",
            "Keys: 1-4=Select Action | t=Toggle Auto-cycle | u=Undo | c=Clear | d=Delete | s=Save | q=Quit"
        ]
        
        y = 30
        for text in instructions:
            # Background rectangle
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(self.display_image, (5, y - 20), 
                         (15 + text_width, y + 5), (0, 0, 0), -1)
            
            # Text
            cv2.putText(self.display_image, text, (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 25
    
    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        """Handle mouse events"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add vertex
            self.current_polygon.append((x, y))
            logger.info(f"Added vertex at ({x}, {y})")
            self._redraw()
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Complete polygon and assign action
            if len(self.current_polygon) >= 3:
                self._complete_polygon()
            else:
                print("Need at least 3 vertices to complete polygon")
    
    def _complete_polygon(self) -> None:
        """Complete current polygon and show action selection dialog"""
        if len(self.current_polygon) < 3:
            return
        
        # Show action selection
        print("\n" + "="*50)
        print("Select action for this polygon:")
        for idx, action in enumerate(ACTIONS, 1):
            marker = "→" if idx - 1 == self.current_action_idx else " "
            print(f"{marker} {idx}. {action}")
        print("="*50)
        
        # Use current action
        selected_action = ACTIONS[self.current_action_idx]
        
        # Save polygon
        zone = {
            "action": selected_action,
            "points": self.current_polygon.copy()
        }
        self.completed_polygons.append(zone)
        
        logger.info(f"Completed polygon with action: {selected_action}")
        print(f"✓ Created {selected_action} zone with {len(self.current_polygon)} vertices")
        
        # Clear current polygon
        self.current_polygon = []
        
        # Move to next action only if auto-cycle is enabled
        if self.auto_cycle_actions:
            self.current_action_idx = (self.current_action_idx + 1) % len(ACTIONS)
            print(f"  → Auto-cycled to: {ACTIONS[self.current_action_idx]}")
        else:
            print(f"  → Still on: {ACTIONS[self.current_action_idx]} (press 1-4 to change)")
        
        self._redraw()
    
    def _undo_vertex(self) -> None:
        """Undo last vertex"""
        if len(self.current_polygon) > 0:
            removed = self.current_polygon.pop()
            logger.info(f"Removed vertex at {removed}")
            self._redraw()
    
    def _clear_current(self) -> None:
        """Clear current polygon"""
        if len(self.current_polygon) > 0:
            self.current_polygon = []
            logger.info("Cleared current polygon")
            self._redraw()
    
    def _delete_last_polygon(self) -> None:
        """Delete last completed polygon"""
        if len(self.completed_polygons) > 0:
            removed = self.completed_polygons.pop()
            logger.info(f"Deleted polygon: {removed['action']}")
            print(f"✓ Deleted last {removed['action']} zone")
            self._redraw()
    
    def _cycle_action(self) -> None:
        """Cycle through actions"""
        self.current_action_idx = (self.current_action_idx + 1) % len(ACTIONS)
        logger.info(f"Switched to action: {ACTIONS[self.current_action_idx]}")
        self._redraw()
    
    def run(self) -> None:
        """Run the interactive polygon drawer"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        self._redraw()
        
        print("\n" + "="*60)
        print("ACTION ZONE DRAWER")
        print("="*60)
        print("\nInstructions:")
        print("  • Left click to add polygon vertices")
        print("  • Right click to complete polygon")
        print("  • Press number keys (1-5) to select action:")
        for idx, action in enumerate(ACTIONS, 1):
            current = " ← Current" if idx - 1 == self.current_action_idx else ""
            print(f"      {idx} = {action}{current}")
        print("\nOther keys:")
        print("  • t: Toggle auto-cycle (OFF by default - stay on same action)")
        print("  • u: Undo last vertex")
        print("  • c: Clear current polygon")
        print("  • d: Delete last completed polygon")
        print("  • s: Save to JSON")
        print("  • q or ESC: Quit")
        print("\nTip: You can draw multiple polygons for the same action!")
        print("     Just keep drawing without pressing 1-5 to change action.")
        print("="*60 + "\n")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # q or ESC
                break
            elif key == ord('u'):
                self._undo_vertex()
            elif key == ord('c'):
                self._clear_current()
            elif key == ord('d'):
                self._delete_last_polygon()
            elif key == ord('s'):
                self._save_zones()
            elif key == ord('t'):
                # Toggle auto-cycle
                self.auto_cycle_actions = not self.auto_cycle_actions
                status = "ON" if self.auto_cycle_actions else "OFF"
                print(f"Auto-cycle actions: {status}")
                logger.info(f"Auto-cycle: {status}")
                self._redraw()
            elif key == ord('a'):
                self._cycle_action()
            elif ord('1') <= key <= ord('5'):
                # Select action by number
                self.current_action_idx = key - ord('1')
                logger.info(f"Selected action: {ACTIONS[self.current_action_idx]}")
                print(f"Selected action: {ACTIONS[self.current_action_idx]}")
                self._redraw()
        
        # Save before quitting
        if len(self.completed_polygons) > 0:
            response = input("\nSave zones before quitting? (y/n): ")
            if response.lower() == 'y':
                self._save_zones()
        
        cv2.destroyAllWindows()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Draw action zones on BEV overlay")
    parser.add_argument(
        "--image",
        type=str,
        default="data/chicken_multiview/bev_overlays/00000.png",
        help="Path to BEV overlay image"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/action_zones.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="Display scale factor (default: 2.0 = 200%% larger)"
    )
    
    args = parser.parse_args()
    
    try:
        drawer = PolygonDrawer(args.image, args.output, display_scale=args.scale)
        drawer.run()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
