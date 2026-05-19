"""BEV (Bird's Eye View) canvas for 3D box placement"""

from __future__ import annotations

import math
from typing import Optional

import imgviz
import numpy as np
from numpy.typing import NDArray
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import QPointF, Qt
from loguru import logger

from argustrack.shape import Shape
from argustrack.utils.constants import DEFAULT_BEV_X, DEFAULT_BEV_Y

# Use same color map as main app for consistency
LABEL_COLORMAP: NDArray[np.uint8] = imgviz.label_colormap()


class BEVCanvas(QtWidgets.QWidget):
    """Canvas for BEV view to place 3D boxes"""
    
    boxPlaced = QtCore.pyqtSignal(float, float, float, float, float, float)  # x, y, z, w, h, d
    boxSelected = QtCore.pyqtSignal(int)  # box index
    boxMoved = QtCore.pyqtSignal(int, float, float, float)  # box index, new x, new y, new z
    boxSizeChanged = QtCore.pyqtSignal(int, float, float, float)  # box index, new w, new h, new d
    boxDeleted = QtCore.pyqtSignal(int, str, object)  # box index, label, group_id
    
    # Point signals
    pointPlaced = QtCore.pyqtSignal(float, float, object)  # x, y, group_id
    pointSelected = QtCore.pyqtSignal(object)  # group_id (for highlighting in camera views)
    pointHovered = QtCore.pyqtSignal(object)  # group_id (for highlighting in camera views)
    pointDoubleClicked = QtCore.pyqtSignal(int, object)  # point_idx, group_id (for editing)
    pointDeleted = QtCore.pyqtSignal(object)  # group_id
    pointMoved = QtCore.pyqtSignal(object, float, float)  # group_id, new_x, new_y
    pointLocked = QtCore.pyqtSignal(object, bool)  # group_id, locked (True = locked, False = unlocked)
    mousePositionChanged = QtCore.pyqtSignal(float, float)  # x, y (grid coordinates) - for real-time projection
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)  # Enable keyboard focus
        
        # Color shift for auto shape colors (must match app.py config)
        self._shift_auto_shape_color: int = 0
        
        # BEV view settings
        # Grid dimensions from constants (pixels in grid space)
        self.grid_width = DEFAULT_BEV_X  # Grid X dimension (1200 pixels)
        self.grid_height = DEFAULT_BEV_Y  # Grid Y dimension (800 pixels)
        
        # Store the actual drawing area (accounting for aspect ratio maintenance)
        self.draw_offset_x = 0  # X offset for letterboxing
        self.draw_offset_y = 0  # Y offset for pillarboxing
        self.draw_width = 0  # Actual drawing width
        self.draw_height = 0  # Actual drawing height
        
        # Legacy fields (kept for compatibility but not used with grid scaling)
        self.bev_scale = 1.0  # pixels per meter (legacy)
        self.bev_offset_x = 0.0  # offset in meters (legacy)
        self.bev_offset_y = 0.0  # offset in meters (legacy)
        self.bev_width = 50.0  # meters (legacy)
        self.bev_height = 50.0  # meters (legacy)
        
        # 3D boxes: list of (center, size, rotation, label, group_id)
        self.boxes_3d: list[tuple] = []
        self.selected_box_idx: Optional[int] = None
        self.hovered_box_idx: Optional[int] = None
        
        # Points: list of (x, y, label, group_id) - for simple markers on BEV
        self.points: list[tuple] = []
        self.selected_point_idx: Optional[int] = None
        self.hovered_point_idx: Optional[int] = None
        self.point_radius = 4  # radius in pixels
        
        # Track locked state for each point by group_id
        # Locked points are "locked label" - cannot be edited and shown in black color
        self.points_locked: dict[Optional[int], bool] = {}  # group_id -> locked (True/False)
        
        # Track action for each point by group_id
        self.points_actions: dict[Optional[int], str] = {}  # group_id -> action (walking, eating, sitting, standing)
        
        # Dragging state for points
        self.dragging_point_idx: Optional[int] = None
        self.drag_start_pos: Optional[QPointF] = None
        # Minimum distance to consider as dragging (in screen pixels)
        self.drag_threshold = 5.0  # pixels
        
        # Drawing state
        self.drawing_box = False
        self.drawing_start_pos: Optional[QPointF] = None
        # Default 3D box size in meters: width, height, depth
        self.current_box_size = [20.0, 20.0, 20.0]
        
        # Movement step size (meters)
        # These will typically be overridden by the control in the BEV dock,
        # but set sensible defaults here as well.
        self.move_step = 15.0
        self.move_step_large = 150.0  # Large step size (with Shift)
        
        # Camera overlay data
        self.camera_calibrations: dict = {}  # camera_id -> calibration
        self.camera_data: list = []  # List of camera data dicts
        # Whether to draw camera positions/FOV on top of BEV background
        self.show_camera_overlay: bool = True
        
        # Colors - make them more transparent so BEV overlay is clearly visible
        self.grid_color = QtGui.QColor(200, 200, 200, 60)  # Very light grid
        self.box_color = QtGui.QColor(0, 255, 0)
        self.selected_box_color = QtGui.QColor(255, 0, 0)
        self.hovered_box_color = QtGui.QColor(255, 255, 0)
        self.camera_color = QtGui.QColor(0, 150, 255)  # Blue for cameras
        # Used only if we ever re-enable FOV fill; for now we draw outline only
        self.camera_fov_color = QtGui.QColor(0, 150, 255, 16)

        # Optional BEV background image (e.g. stitched top-down view)
        self.background_image: QtGui.QImage | None = None
        # Use full opacity so BEV overlay is clearly visible
        self.background_alpha: float = 1.0
    
    def setMoveStep(self, step: float):
        """Set movement step size"""
        self.move_step = step
        self.move_step_large = step * 10.0
    
    def setColorShift(self, shift: int):
        """
        Set color shift for auto shape colors to match app.py config.
        
        Args:
            shift: shift_auto_shape_color value from config
        """
        self._shift_auto_shape_color = shift
        self.update()
    
    def _get_rgb_by_group_id(self, group_id: int) -> tuple[int, int, int]:
        """
        Get RGB color for a group_id using the same colormap as app.py.
        This ensures BEV points have the same color as camera bounding boxes.
        
        Args:
            group_id: Group ID for the shape
            
        Returns:
            RGB tuple (r, g, b) with values 0-255
        """
        color_id: int = (
            1  # skip black color by default
            + group_id
            + self._shift_auto_shape_color
        )
        rgb: tuple[int, int, int] = tuple(
            LABEL_COLORMAP[color_id % len(LABEL_COLORMAP)].tolist()
        )
        return rgb

    def setBackgroundImage(self, image: QtGui.QImage | str, alpha: float = 0.8) -> None:
        """Set BEV background image (top-down overlay) and optional opacity."""
        if isinstance(image, str):
            img = QtGui.QImage(image)
            if img.isNull():
                logger.warning(f"Failed to load BEV background image: {image}")
                return
            self.background_image = img
        else:
            self.background_image = image

        # Clamp alpha between 0 and 1
        self.background_alpha = float(max(0.0, min(alpha, 1.0)))
        self.update()
    
    def setCameraOverlay(self, camera_calibrations: dict, camera_data: list):
        """Set camera calibrations and data for overlay"""
        self.camera_calibrations = camera_calibrations
        self.camera_data = camera_data
        self.update()
    
    def setBEVParams(self, width: float, height: float, scale: float = 1.0):
        """
        Set BEV view parameters (legacy method for compatibility).
        
        Note: This method is kept for backward compatibility but the grid scaling
        is now based on DEFAULT_BEV_X and DEFAULT_BEV_Y from constants.
        
        Args:
            width: BEV width (legacy, not used)
            height: BEV height (legacy, not used)
            scale: BEV scale (legacy, not used)
        """
        logger.warning("setBEVParams is deprecated. Grid scaling now uses constants.DEFAULT_BEV_X and DEFAULT_BEV_Y")
        self.bev_width = width
        self.bev_height = height
        self.bev_scale = scale
        self.update()
    
    def setGridDimensions(self, grid_width: int, grid_height: int):
        """
        Update grid dimensions for coordinate transformation.
        
        Grid dimensions define the logical coordinate space and should remain constant.
        When the widget is resized, the screen coordinates change, but grid coordinates stay the same.
        
        Args:
            grid_width: Grid width in pixels (e.g., DEFAULT_BEV_X = 1200)
            grid_height: Grid height in pixels (e.g., DEFAULT_BEV_Y = 800)
        """
        logger.info(f"Setting grid dimensions to {grid_width} x {grid_height}")
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.update()
    
    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """Handle resize event - maintain aspect ratio and adjust coordinate system"""
        super().resizeEvent(event)
        old_size = event.oldSize()
        new_size = event.size()
        
        # Calculate the drawable area maintaining aspect ratio
        self._calculate_drawable_area()
        
        if old_size.width() > 0 and old_size.height() > 0:
            logger.debug(
                f"BEV Canvas resized: {old_size.width()}x{old_size.height()} → "
                f"{new_size.width()}x{new_size.height()} "
                f"(grid: {self.grid_width}x{self.grid_height}, "
                f"draw: {self.draw_width}x{self.draw_height})"
            )
        
        # Update display
        self.update()
    
    def _calculate_drawable_area(self) -> None:
        """
        Calculate the actual drawable area maintaining aspect ratio.
        
        When the widget aspect ratio doesn't match grid aspect ratio,
        we add letterboxing (black bars on sides) or pillarboxing (black bars top/bottom).
        """
        widget_width = self.width()
        widget_height = self.height()
        
        # Calculate aspect ratios
        grid_aspect = self.grid_width / self.grid_height if self.grid_height > 0 else 1.0
        widget_aspect = widget_width / widget_height if widget_height > 0 else 1.0
        
        if widget_aspect > grid_aspect:
            # Widget is wider than grid - add letterboxing (vertical bars on sides)
            self.draw_height = widget_height
            self.draw_width = int(widget_height * grid_aspect)
            self.draw_offset_x = (widget_width - self.draw_width) // 2
            self.draw_offset_y = 0
        else:
            # Widget is taller than grid - add pillarboxing (horizontal bars on top/bottom)
            self.draw_width = widget_width
            self.draw_height = int(widget_width / grid_aspect)
            self.draw_offset_x = 0
            self.draw_offset_y = (widget_height - self.draw_height) // 2
    
    def addBox3D(self, x: float, y: float, z: float, w: float, h: float, d: float,
                 label: str = "", group_id: Optional[int] = None, rotation: float = 0.0):
        """Add a 3D box"""
        self.boxes_3d.append((np.array([x, y, z]), np.array([w, h, d]), rotation, label, group_id))
        self.update()
    
    def clearBoxes(self):
        """Clear all 3D boxes"""
        self.boxes_3d = []
        self.selected_box_idx = None
        self.hovered_box_idx = None
        self.update()
    
    def addPoint(self, x: float, y: float, label: str = "object", group_id: Optional[int] = None):
        """Add a point marker to the BEV canvas. If a point with the same group_id already exists, update it instead of adding a duplicate."""
        # Check if point with same group_id already exists
        if group_id is not None:
            for i, (px, py, plabel, pgid) in enumerate(self.points):
                if pgid == group_id:
                    # Update existing point instead of adding duplicate
                    self.points[i] = (x, y, label, group_id)
                    self.update()
                    return
        
        # No existing point with this group_id, add new point
        self.points.append((x, y, label, group_id))
        self.update()
    
    def clearPoints(self):
        """Clear all points"""
        self.points = []
        self.points_locked = {}
        self.points_actions = {}
        self.selected_point_idx = None
        self.hovered_point_idx = None
        self.update()
    
    def deletePointByGroupId(self, group_id: int) -> bool:
        """Delete a point by group_id"""
        for i, (x, y, label, gid) in enumerate(self.points):
            if gid == group_id:
                self.points.pop(i)
                # Remove locked state if exists
                if group_id in self.points_locked:
                    del self.points_locked[group_id]
                # Remove action if exists
                if group_id in self.points_actions:
                    del self.points_actions[group_id]
                if self.selected_point_idx == i:
                    self.selected_point_idx = None
                elif self.selected_point_idx is not None and self.selected_point_idx > i:
                    self.selected_point_idx -= 1
                self.update()
                return True
        return False
    
    def setPointLocked(self, group_id: Optional[int], locked: bool, emit_signal: bool = True):
        """Set locked state for a point by group_id
        
        Args:
            group_id: Group ID of the point
            locked: True to lock, False to unlock
            emit_signal: If True, emit pointLocked signal (default True)
        """
        if group_id is None:
            return
        self.points_locked[group_id] = locked
        self.update()
        # Emit signal to notify (unless loading from file)
        if emit_signal:
            self.pointLocked.emit(group_id, locked)
    
    def isPointLocked(self, group_id: Optional[int]) -> bool:
        """Check if a point is locked by group_id"""
        if group_id is None:
            return False
        return self.points_locked.get(group_id, False)
    
    def setPointAction(self, group_id: Optional[int], action: str):
        """Set action for a point by group_id
        
        Args:
            group_id: Group ID of the point
            action: Action name (walking, eating, sitting, standing)
        """
        if group_id is None:
            return
        self.points_actions[group_id] = action
        self.update()
    
    def getPointAction(self, group_id: Optional[int]) -> str:
        """Get action for a point by group_id
        
        Returns:
            Action name, defaults to "walking" if not set
        """
        if group_id is None:
            return "walking"
        return self.points_actions.get(group_id, "walking")

    def updatePoint(self, point_idx: int, x: Optional[float] = None, y: Optional[float] = None, 
                   label: Optional[str] = None, group_id: Optional[int] = None) -> bool:
        """Update point data by index"""
        if not (0 <= point_idx < len(self.points)):
            return False
        
        old_x, old_y, old_label, old_gid = self.points[point_idx]
        new_x = x if x is not None else old_x
        new_y = y if y is not None else old_y
        new_label = label if label is not None else old_label
        new_gid = group_id if group_id is not None else old_gid
        
        logger.info(f"BEV updatePoint: old_gid={old_gid}, new_gid={new_gid}, shift={self._shift_auto_shape_color}")
        self.points[point_idx] = (new_x, new_y, new_label, new_gid)
        self.update()
        return True

    def getPointByGroupId(self, group_id: int) -> Optional[tuple]:
        """Get point data by group_id"""
        for i, (x, y, label, gid) in enumerate(self.points):
            if gid == group_id:
                return (i, x, y, label, gid)
        return None
    
    def _grid_to_screen(self, grid_x: float, grid_y: float) -> QPointF:
        """
        Convert grid coordinates (logical/mem coordinates) to screen coordinates (display).
        
        COORDINATE SYSTEM:
        - Grid space (logical): Fixed (0,0) to (grid_width, grid_height) - e.g., (0,0) to (1200,800)
        - Screen space (display): Variable (0,0) to (widget.width(), widget.height()) - changes with resize
        - Drawable area: The actual area where content is drawn (maintains aspect ratio)
        
        The drawable area may be smaller than the widget due to letterboxing/pillarboxing
        to maintain the correct aspect ratio.
        
        Example with letterboxing (widget wider than needed):
        - Grid coords (600, 400) at center of 1200x800 grid
        - Widget is 1000x600, but drawable area is 900x600 centered at offset (50, 0)
        - Screen coords = (50, 0) + (600 * 900/1200, 400 * 600/800) = (500, 300)
        
        Args:
            grid_x, grid_y: Coordinates in grid space (0 to grid_width, 0 to grid_height)
            
        Returns:
            QPointF with screen coordinates (includes drawable area offset)
        """
        try:
            # Ensure drawable area is calculated
            if self.draw_width == 0 or self.draw_height == 0:
                self._calculate_drawable_area()
            
            # Scale from grid space to drawable area
            scale_x = self.draw_width / self.grid_width if self.grid_width > 0 else 1.0
            scale_y = self.draw_height / self.grid_height if self.grid_height > 0 else 1.0
            
            # Convert and add offset for letterboxing/pillarboxing
            screen_x = self.draw_offset_x + (grid_x * scale_x)
            screen_y = self.draw_offset_y + (grid_y * scale_y)
            
            return QPointF(screen_x, screen_y)
        except Exception as e:
            logger.error(f"Error in _grid_to_screen({grid_x}, {grid_y}): {e}")
            return QPointF(0, 0)
    
    def _screen_to_grid(self, screen_pos: QPointF) -> tuple[float, float]:
        """
        Convert screen coordinates (display) to grid coordinates (logical/mem coordinates).
        
        COORDINATE SYSTEM:
        - Screen space (display): Variable (0,0) to (widget.width(), widget.height()) - changes with resize
        - Drawable area: The actual area where content is drawn (maintains aspect ratio)
        - Grid space (logical): Fixed (0,0) to (grid_width, grid_height) - e.g., (0,0) to (1200,800)
        
        The drawable area may be smaller than the widget due to letterboxing/pillarboxing.
        Clicks outside the drawable area are clamped to the nearest edge.
        
        Example with letterboxing:
        - Click at screen (500, 300), widget is 1000x600, drawable area is 900x600 at offset (50, 0)
        - Relative to drawable: (500-50, 300-0) = (450, 300)
        - Grid coords = (450 * 1200/900, 300 * 800/600) = (600, 400)
        
        Args:
            screen_pos: Position in screen/canvas coordinates
            
        Returns:
            (grid_x, grid_y): Coordinates in grid space (logical coordinates for storage)
        """
        try:
            # Ensure drawable area is calculated
            if self.draw_width == 0 or self.draw_height == 0:
                self._calculate_drawable_area()
            
            # Subtract offset to get position relative to drawable area
            relative_x = screen_pos.x() - self.draw_offset_x
            relative_y = screen_pos.y() - self.draw_offset_y
            
            # Scale from drawable area to grid space
            scale_x = self.grid_width / self.draw_width if self.draw_width > 0 else 1.0
            scale_y = self.grid_height / self.draw_height if self.draw_height > 0 else 1.0
            
            grid_x = relative_x * scale_x
            grid_y = relative_y * scale_y
            
            # Clamp to valid grid range (in case click was outside drawable area)
            grid_x = max(0, min(self.grid_width, grid_x))
            grid_y = max(0, min(self.grid_height, grid_y))
            
            return grid_x, grid_y
        except Exception as e:
            logger.error(f"Error in _screen_to_grid({screen_pos.x()}, {screen_pos.y()}): {e}")
            return 0.0, 0.0
    
    def _get_box_at_position(self, pos: QPointF) -> Optional[int]:
        """Get box index at given screen position"""
        for idx, (center, size, rotation, _, _) in enumerate(self.boxes_3d):
            screen_center = self._grid_to_screen(center[0], center[1])
            # Calculate screen size based on canvas/grid scaling
            canvas_width = self.width()
            scale_x = canvas_width / self.grid_width
            w_screen = size[0] * scale_x  # width in screen pixels
            d_screen = size[2] * scale_x  # depth in screen pixels
            
            # Simple rectangle check (ignoring rotation for now)
            dx = pos.x() - screen_center.x()
            dy = pos.y() - screen_center.y()
            
            if abs(dx) <= w_screen / 2 and abs(dy) <= d_screen / 2:
                return idx
        
        return None

    def _get_point_at_position(self, pos: QPointF) -> Optional[int]:
        """Get point index at given screen position"""
        for idx, (x, y, label, group_id) in enumerate(self.points):
            screen_pos = self._grid_to_screen(x, y)
            
            # Check if click is within point radius
            dx = pos.x() - screen_pos.x()
            dy = pos.y() - screen_pos.y()
            distance = math.sqrt(dx * dx + dy * dy)
            
            if distance <= self.point_radius + 2:  # +2 for easier clicking
                return idx
        
        return None
    
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Paint BEV view with proper aspect ratio maintenance"""
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # Ensure drawable area is calculated
        if self.draw_width == 0 or self.draw_height == 0:
            self._calculate_drawable_area()
        
        # Fill entire widget with black (for letterboxing/pillarboxing areas)
        p.fillRect(self.rect(), QtGui.QColor(0, 0, 0))
        
        # Define the drawable rectangle (maintains aspect ratio)
        draw_rect = QtCore.QRect(
            self.draw_offset_x,
            self.draw_offset_y,
            self.draw_width,
            self.draw_height
        )
        
        # Draw background in drawable area (BEV overlay if available, otherwise gray)
        if self.background_image is not None and not self.background_image.isNull():
            pixmap = QtGui.QPixmap.fromImage(self.background_image)
            
            # Scale to fit drawable area exactly (maintaining aspect ratio)
            scaled = pixmap.scaled(
                draw_rect.size(),
                Qt.KeepAspectRatio,  # Changed from KeepAspectRatioByExpanding
                Qt.SmoothTransformation,
            )
            
            # Center the scaled image in the drawable rect if there's any size mismatch
            draw_x = self.draw_offset_x + (self.draw_width - scaled.width()) // 2
            draw_y = self.draw_offset_y + (self.draw_height - scaled.height()) // 2

            p.save()
            p.setOpacity(self.background_alpha)
            p.drawPixmap(draw_x, draw_y, scaled)
            p.restore()
        else:
            # Fallback: flat dark gray background in drawable area
            p.fillRect(draw_rect, QtGui.QColor(30, 30, 30))
        
        # Draw grid - only within visible widget bounds to avoid overflow.
        # Keep it very subtle so BEV overlay is clearly visible.
        p.setPen(QtGui.QPen(self.grid_color, 1))
        grid_spacing = 5.0  # meters
        grid_spacing_screen = grid_spacing * self.bev_scale
        
        widget_width = self.width()
        widget_height = self.height()
        center_x = widget_width / 2
        center_y = widget_height / 2
        
        # Only draw grid within widget bounds with small margin
        margin = 100  # pixels
        visible_left = -margin
        visible_right = widget_width + margin
        visible_top = -margin
        visible_bottom = widget_height + margin
        
        # Helper function to safely clamp to int32 range
        def safe_int(value):
            return max(-2147483648, min(int(value), 2147483647))
        
        # Vertical lines - only draw lines that intersect visible area
        if grid_spacing_screen > 0:
            # Find first grid line that's within or before visible area
            y_start_world = (visible_top - center_y) / self.bev_scale + self.bev_offset_y
            y_end_world = (visible_bottom - center_y) / self.bev_scale + self.bev_offset_y
            
            # Start from a grid line near the visible area
            grid_y_world = (y_start_world // grid_spacing) * grid_spacing
            max_iterations = int((y_end_world - y_start_world) / grid_spacing) + 10
            
            iteration = 0
            while grid_y_world <= y_end_world and iteration < max_iterations:
                y_screen = center_y + (grid_y_world - self.bev_offset_y) * self.bev_scale
                
                # Only draw if within visible bounds
                if visible_top <= y_screen <= visible_bottom:
                    y_int = safe_int(y_screen)
                    start_x_int = safe_int(visible_left)
                    end_x_int = safe_int(visible_right)
                    
                    if abs(start_x_int) < 2000000000 and abs(end_x_int) < 2000000000:
                        p.drawLine(start_x_int, y_int, end_x_int, y_int)
                
                grid_y_world += grid_spacing
                iteration += 1
        
        # Horizontal lines - only draw lines that intersect visible area
        if grid_spacing_screen > 0:
            # Find first grid line that's within or before visible area
            x_start_world = (visible_left - center_x) / self.bev_scale + self.bev_offset_x
            x_end_world = (visible_right - center_x) / self.bev_scale + self.bev_offset_x
            
            # Start from a grid line near the visible area
            grid_x_world = (x_start_world // grid_spacing) * grid_spacing
            max_iterations = int((x_end_world - x_start_world) / grid_spacing) + 10
            
            iteration = 0
            while grid_x_world <= x_end_world and iteration < max_iterations:
                x_screen = center_x + (grid_x_world - self.bev_offset_x) * self.bev_scale
                
                # Only draw if within visible bounds
                if visible_left <= x_screen <= visible_right:
                    x_int = safe_int(x_screen)
                    start_y_int = safe_int(visible_top)
                    end_y_int = safe_int(visible_bottom)
                    
                    if abs(start_y_int) < 2000000000 and abs(end_y_int) < 2000000000:
                        p.drawLine(x_int, start_y_int, x_int, end_y_int)
                
                grid_x_world += grid_spacing
                iteration += 1
        
        # Draw origin with coordinate labels
        origin = self._grid_to_screen(0, 0)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0), 2))
        p.drawLine(int(origin.x() - 10), int(origin.y()), int(origin.x() + 10), int(origin.y()))
        p.drawLine(int(origin.x()), int(origin.y() - 10), int(origin.x()), int(origin.y() + 10))
        
        # Draw coordinate labels at origin
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))
        # p.drawText(int(origin.x() + 12), int(origin.y() - 12), "Origin (0, 0)")
        
        # Draw coordinate axes labels
        # p.drawText(int(origin.x() + 12), int(origin.y() + 5), "X →")
        # p.drawText(int(origin.x() - 25), int(origin.y() - 12), "↑ Y")
        
        # Draw camera overlays (positions and FOV)
        # if self.show_camera_overlay and self.camera_calibrations and self.camera_data:
        #     for cam_data in self.camera_data:
        #         camera_id = cam_data.get("camera_id")
        #         calibration = self.camera_calibrations.get(camera_id)
                
        #         if not calibration:
        #             continue
                
        #         # Get camera position in world space
        #         R = calibration.extrinsic[:3, :3]
        #         t = calibration.extrinsic[:3, 3]
        #         camera_pos_world = -R.T @ t
                
        #         # Draw camera position only (FOV removed for cleaner view)
        #         cam_screen = self._grid_to_screen(camera_pos_world[0], camera_pos_world[1])
        #         p.setPen(QtGui.QPen(self.camera_color, 3))
        #         p.setBrush(QtGui.QBrush(self.camera_color, Qt.SolidPattern))
        #         p.drawEllipse(int(cam_screen.x() - 5), int(cam_screen.y() - 5), 10, 10)
        
        # Draw 3D boxes (top-down view, showing w and d)
        for idx, (center, size, rotation, label, group_id) in enumerate(self.boxes_3d):
            screen_center = self._grid_to_screen(center[0], center[1])
            # Calculate screen size based on canvas/grid scaling
            canvas_width = self.width()
            scale_x = canvas_width / self.grid_width
            w_screen = size[0] * scale_x  # width in screen pixels
            d_screen = size[2] * scale_x  # depth in screen pixels
            
            # Determine color
            if idx == self.selected_box_idx:
                color = self.selected_box_color
            elif idx == self.hovered_box_idx:
                color = self.hovered_box_color
            else:
                color = self.box_color
            
            p.setPen(QtGui.QPen(color, 2))
            p.setBrush(QtGui.QBrush(color, Qt.SolidPattern))
            
            # Draw rectangle (top-down view: w x d, small size)
            rect = QtCore.QRectF(
                screen_center.x() - w_screen / 2,
                screen_center.y() - d_screen / 2,
                w_screen,
                d_screen
            )
            p.drawRect(rect)
            
            # Draw label if available
            if label:
                p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
                p.drawText(rect, Qt.AlignCenter, label)
        
        # Draw points (simple markers)
        for idx, (x, y, label, group_id) in enumerate(self.points):
            screen_pos = self._grid_to_screen(x, y)
            
            # Check if point is locked - locked points are always black
            is_locked = self.isPointLocked(group_id)
            
            if is_locked:
                # Locked points are drawn in black
                color = QtGui.QColor(0, 0, 0)  # Black color for locked points
            else:
                # Determine color based on group_id for consistency with camera views
                if group_id is not None:
                    # Use same colormap as app.py for consistent colors
                    r, g, b = self._get_rgb_by_group_id(group_id)
                    color = QtGui.QColor(r, g, b)
                    # Debug: log color for first point only to avoid spam
                    # if idx == 0:
                        # logger.debug(f"BEV paintEvent: point[{idx}] group_id={group_id}, color=RGB({r},{g},{b})")
                else:
                    color = self.box_color
                
                # Check if selected or hovered (override color, but not for locked)
                if idx == self.selected_point_idx:
                    color = self.selected_box_color
                elif idx == self.hovered_point_idx:
                    color = self.hovered_box_color
            
            # Draw filled circle
            p.setPen(QtGui.QPen(color, 2))
            p.setBrush(QtGui.QBrush(color, Qt.SolidPattern))
            p.drawEllipse(screen_pos, self.point_radius, self.point_radius)
            
            # Draw label text next to point
            if group_id is not None:
                p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
                text_rect = QtCore.QRectF(
                    screen_pos.x() + self.point_radius + 2,
                    screen_pos.y() - 8,
                    50, 16
                )
                p.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, str(group_id))
        
        # Draw current drawing box
        if self.drawing_box and self.drawing_start_pos:
            current_pos = self.drawing_start_pos
            x, y = self._screen_to_grid(current_pos)
            center_x_grid, center_y_grid = self._screen_to_grid(self.drawing_start_pos)
            
            # Draw preview box (top-down view: w x d)
            screen_center = self._grid_to_screen(center_x_grid, center_y_grid)
            # Calculate screen size based on canvas/grid scaling
            canvas_width = self.width()
            scale_x = canvas_width / self.grid_width
            w_screen = self.current_box_size[0] * scale_x  # width in screen pixels
            d_screen = self.current_box_size[2] * scale_x  # depth in screen pixels
            
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 2, Qt.DashLine))
            p.setBrush(QtCore.Qt.NoBrush)
            rect = QtCore.QRectF(
                screen_center.x() - w_screen / 2,
                screen_center.y() - d_screen / 2,
                w_screen,
                d_screen
            )
            p.drawRect(rect)
        
        p.end()
    
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse press"""
        pos = event.localPos()
        
        if event.button() == Qt.LeftButton:
            # First check if clicking on existing point - start dragging
            point_idx = self._get_point_at_position(pos)
            if point_idx is not None:
                _, _, _, group_id = self.points[point_idx]
                # Check if point is locked - locked points cannot be dragged
                if self.isPointLocked(group_id):
                    # Point is locked, don't allow dragging
                    self.selected_point_idx = point_idx
                    self.pointSelected.emit(group_id)
                    self.update()
                    return
                
                self.selected_point_idx = point_idx
                self.dragging_point_idx = point_idx  # Start dragging
                self.drag_start_pos = pos
                self.pointSelected.emit(group_id)
                self.update()
                return
            
            # Then check if clicking on existing box
            box_idx = self._get_box_at_position(pos)
            if box_idx is not None:
                self.selected_box_idx = box_idx
                self.boxSelected.emit(box_idx)
                self.update()
            else:
                # Place a new point at click location
                x, y = self._screen_to_grid(pos)
                self.pointPlaced.emit(x, y, None)  # group_id will be assigned by MainWindow
                self.update()
        elif event.button() == Qt.RightButton:
            # Right-click on point to show context menu
            point_idx = self._get_point_at_position(pos)
            if point_idx is not None:
                self.selected_point_idx = point_idx
                _, _, _, group_id = self.points[point_idx]
                self.pointSelected.emit(group_id)
                
                # Show context menu
                self._show_point_context_menu(event.globalPos(), group_id)
                self.update()
                return
            
            # Right-click to edit box
            box_idx = self._get_box_at_position(pos)
            if box_idx is not None:
                self.selected_box_idx = box_idx
                self.boxSelected.emit(box_idx)
                self.update()
    
    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle double-click to edit point or box"""
        if event.button() == Qt.LeftButton:
            pos = event.localPos()
            
            # First check points
            point_idx = self._get_point_at_position(pos)
            if point_idx is not None:
                _, _, _, group_id = self.points[point_idx]
                # Check if point is locked - locked points cannot be edited
                if self.isPointLocked(group_id):
                    # Point is locked, don't allow editing
                    return
                
                self.selected_point_idx = point_idx
                self.pointDoubleClicked.emit(point_idx, group_id)
                self.update()
                return
            
            # Then check boxes
            box_idx = self._get_box_at_position(pos)
            if box_idx is not None:
                self.selected_box_idx = box_idx
                self.boxSelected.emit(box_idx)
                # Emit signal to open edit dialog
                center, size, rotation, label, group_id = self.boxes_3d[box_idx]
                self.boxSizeChanged.emit(box_idx, size[0], size[1], size[2])
                self.update()
    
    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse move"""
        pos = event.localPos()
        
        # Handle point dragging
        if self.dragging_point_idx is not None and 0 <= self.dragging_point_idx < len(self.points):
            _, _, _, group_id = self.points[self.dragging_point_idx]
            # Check if point is locked - locked points cannot be dragged
            if self.isPointLocked(group_id):
                # Stop dragging if point is locked
                self.dragging_point_idx = None
                self.drag_start_pos = None
                return
            
            # Check if mouse has moved enough to be considered dragging
            if self.drag_start_pos is not None:
                dx = pos.x() - self.drag_start_pos.x()
                dy = pos.y() - self.drag_start_pos.y()
                distance = math.sqrt(dx * dx + dy * dy)
                
                # Only emit pointMoved if mouse has moved beyond threshold
                # This prevents accidental movement when just clicking
                if distance < self.drag_threshold:
                    # Not enough movement, don't emit pointMoved signal
                    # Just update hover state
                    return
            
            # Get grid coordinates for new position
            new_x, new_y = self._screen_to_grid(pos)
            
            # Update point position
            _, _, label, group_id = self.points[self.dragging_point_idx]
            self.points[self.dragging_point_idx] = (new_x, new_y, label, group_id)
            
            # Emit signal during dragging for real-time updates in multiview
            self.pointMoved.emit(group_id, new_x, new_y)
            
            self.update()
            return
        
        # First check point hover (higher priority)
        point_idx = self._get_point_at_position(pos)
        if point_idx != self.hovered_point_idx:
            old_hovered = self.hovered_point_idx
            self.hovered_point_idx = point_idx
            
            # Emit hover signal for highlighting in camera views
            if point_idx is not None:
                _, _, _, group_id = self.points[point_idx]
                self.pointHovered.emit(group_id)
            elif old_hovered is not None:
                # Mouse left point, clear hover highlight
                self.pointHovered.emit(None)
            
            self.update()
        
        # Update hovered box (only if not hovering point)
        if point_idx is None:
            box_idx = self._get_box_at_position(pos)
            if box_idx != self.hovered_box_idx:
                self.hovered_box_idx = box_idx
                self.update()
        
        if self.drawing_box:
            self.update()
        
        # Update tooltip with grid coordinates
        x, y = self._screen_to_grid(pos)
        # self.setToolTip(f"Grid X: {x:.1f}, Grid Y: {y:.1f}")
        
        # Emit mouse position for real-time projection to cameras
        self.mousePositionChanged.emit(x, y)
    
    def leaveEvent(self, event: QtCore.QEvent) -> None:
        """Handle mouse leave event - clear projection markers"""
        # Emit None, None to clear markers
        self.mousePositionChanged.emit(float('nan'), float('nan'))
        super().leaveEvent(event)
    
    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse release"""
        if event.button() == Qt.LeftButton:
            # Finish point dragging
            if self.dragging_point_idx is not None and 0 <= self.dragging_point_idx < len(self.points):
                # Check if mouse actually moved (dragging) or just clicked
                if self.drag_start_pos is not None:
                    pos = event.localPos()
                    dx = pos.x() - self.drag_start_pos.x()
                    dy = pos.y() - self.drag_start_pos.y()
                    distance = math.sqrt(dx * dx + dy * dy)
                    
                    # Only emit pointMoved if mouse moved beyond threshold
                    if distance >= self.drag_threshold:
                        # Get final position and emit signal
                        new_x, new_y, label, group_id = self.points[self.dragging_point_idx]
                        self.pointMoved.emit(group_id, new_x, new_y)
                
                self.dragging_point_idx = None
                self.drag_start_pos = None
                self.update()
                return
            
            # Point placement is now handled in mousePressEvent directly
            if self.drawing_box:
                self.drawing_box = False
                self.drawing_start_pos = None
                self.update()
    
    def setCurrentBoxSize(self, w: float, h: float, d: float):
        """Set current box size for drawing"""
        self.current_box_size = [w, h, d]
    
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Handle keyboard input for moving points/boxes and deleting"""
        # Delete selected point or box with Delete/Backspace
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            # First check if point is selected and valid
            point_idx_to_delete = self.selected_point_idx
            if point_idx_to_delete is not None and 0 <= point_idx_to_delete < len(self.points):
                _, _, _, group_id = self.points[point_idx_to_delete]
                
                # Check if point is locked - locked points cannot be deleted
                if self.isPointLocked(group_id):
                    # Point is locked, don't allow deletion
                    return
                
                # Reset selection state BEFORE emitting signal (signal handler might modify state)
                self.selected_point_idx = None
                self.hovered_point_idx = None
                self.dragging_point_idx = None
                
                # Emit signal so MainWindow can remove all related 2D boxes
                self.pointDeleted.emit(group_id)
                
                # Remove from points (check again in case signal handler already removed it)
                # Find by group_id since index may have changed
                for i, (_, _, _, gid) in enumerate(self.points):
                    if gid == group_id:
                        del self.points[i]
                        break
                
                self.update()
                return
            
            # Then check if box is selected
            if self.selected_box_idx is not None and 0 <= self.selected_box_idx < len(self.boxes_3d):
                center, size, rotation, label, group_id = self.boxes_3d[self.selected_box_idx]
                # Emit signal so MainWindow can remove all projected 2D boxes
                self.boxDeleted.emit(self.selected_box_idx, label, group_id)
                # Remove from BEV
                del self.boxes_3d[self.selected_box_idx]
                self.selected_box_idx = None
                self.hovered_box_idx = None
                self.update()
            else:
                super().keyPressEvent(event)
            return

        # Check for modifier keys
        is_shift = event.modifiers() & Qt.ShiftModifier
        step = self.move_step_large if is_shift else self.move_step

        # Handle point movement with arrow keys
        if self.selected_point_idx is not None and 0 <= self.selected_point_idx < len(self.points):
            x, y, label, group_id = self.points[self.selected_point_idx]
            
            # Check if point is locked - locked points cannot be moved
            if self.isPointLocked(group_id):
                # Point is locked, don't allow movement
                super().keyPressEvent(event)
                return
            
            new_x, new_y = x, y
            moved = False
            
            if event.key() == Qt.Key_Left:
                new_x -= step
                moved = True
            elif event.key() == Qt.Key_Right:
                new_x += step
                moved = True
            elif event.key() == Qt.Key_Up:
                new_y -= step
                moved = True
            elif event.key() == Qt.Key_Down:
                new_y += step
                moved = True
            
            if moved:
                # Update point position
                self.points[self.selected_point_idx] = (new_x, new_y, label, group_id)
                # Emit signal so MainWindow can update camera bounding boxes
                self.pointMoved.emit(group_id, new_x, new_y)
                self.update()
                return
        
        # Handle box movement
        if self.selected_box_idx is None or self.selected_box_idx >= len(self.boxes_3d):
            super().keyPressEvent(event)
            return
        
        # Get current box
        center, size, rotation, label, group_id = self.boxes_3d[self.selected_box_idx]
        
        new_center = center.copy()
        moved = False
        
        if event.key() == Qt.Key_Left:
            new_center[0] -= step
            moved = True
        elif event.key() == Qt.Key_Right:
            new_center[0] += step
            moved = True
        elif event.key() == Qt.Key_Up:
            new_center[1] -= step
            moved = True
        elif event.key() == Qt.Key_Down:
            new_center[1] += step
            moved = True
        elif event.key() == Qt.Key_PageUp:
            new_center[2] += step
            moved = True
        elif event.key() == Qt.Key_PageDown:
            new_center[2] -= step
            moved = True
        
        if moved:
            # Update box position
            self.boxes_3d[self.selected_box_idx] = (new_center, size, rotation, label, group_id)
            
            # Emit signal to update projection
            self.boxMoved.emit(self.selected_box_idx, new_center[0], new_center[1], new_center[2])
            
            self.update()
        else:
            super().keyPressEvent(event)
    
    def getBox3D(self, box_idx: int) -> Optional[tuple]:
        """Get 3D box data by index"""
        if 0 <= box_idx < len(self.boxes_3d):
            return self.boxes_3d[box_idx]
        return None
    
    def updateBox3D(self, box_idx: int, x: float, y: float, z: float, 
                   w: Optional[float] = None, h: Optional[float] = None, d: Optional[float] = None):
        """Update 3D box position and/or size"""
        if 0 <= box_idx < len(self.boxes_3d):
            center, size, rotation, label, group_id = self.boxes_3d[box_idx]
            new_center = np.array([x, y, z])
            new_size = size.copy()
            if w is not None:
                new_size[0] = w
            if h is not None:
                new_size[1] = h
            if d is not None:
                new_size[2] = d
            self.boxes_3d[box_idx] = (new_center, new_size, rotation, label, group_id)
            self.update()
    
    def _show_point_context_menu(self, global_pos: QPointF, group_id: Optional[int]):
        """Show context menu for a point when right-clicked"""
        if group_id is None:
            return
        
        menu = QtWidgets.QMenu(self)
        
        # Check if point is currently locked
        is_locked = self.isPointLocked(group_id)
        
        if is_locked:
            # If locked, show option to unlock
            unlock_action = menu.addAction("Unlock label")
            action = menu.exec_(global_pos)
            
            if action == unlock_action:
                self.setPointLocked(group_id, False)
        else:
            # If not locked, show option to lock
            lock_action = menu.addAction("Lock label")
            action = menu.exec_(global_pos)
            
            if action == lock_action:
                self.setPointLocked(group_id, True)

