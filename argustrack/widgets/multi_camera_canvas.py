from __future__ import annotations

import enum
import math
from typing import NamedTuple

import numpy as np
from loguru import logger
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import QPointF
from PyQt5.QtCore import Qt

from argustrack.shape import Shape
from argustrack.widgets.canvas import Canvas, CanvasMode, CURSOR_DEFAULT, CURSOR_DRAW, CURSOR_GRAB, CURSOR_MOVE, CURSOR_POINT


class CameraCell(NamedTuple):
    """Represents a single camera cell in the grid"""
    camera_id: str
    image_path: str
    image: QtGui.QImage
    pixmap: QtGui.QPixmap
    cell_rect: QtCore.QRectF  # Position in grid
    image_rect: QtCore.QRectF  # Scaled image position within cell
    shapes: list[Shape]
    calibration_path: str | None = None


class MultiCameraCanvas(QtWidgets.QWidget):
    """Canvas widget that displays multiple camera views in a grid layout"""
    
    zoomRequest = QtCore.pyqtSignal(int, QPointF)
    scrollRequest = QtCore.pyqtSignal(int, int)
    newShape = QtCore.pyqtSignal()
    selectionChanged = QtCore.pyqtSignal(list)
    shapeMoved = QtCore.pyqtSignal()
    drawingPolygon = QtCore.pyqtSignal(bool)
    vertexSelected = QtCore.pyqtSignal(bool)
    mouseMoved = QtCore.pyqtSignal(QPointF)
    statusUpdated = QtCore.pyqtSignal(str)

    mode: CanvasMode = CanvasMode.EDIT
    _createMode = "polygon"
    _fill_drawing = False

    def __init__(self, *args, **kwargs):
        self.epsilon = kwargs.pop("epsilon", 10.0)
        self.double_click = kwargs.pop("double_click", "close")
        self.num_backups = kwargs.pop("num_backups", 10)
        self._crosshair = kwargs.pop(
            "crosshair",
            {
                "polygon": False,
                "rectangle": True,
                "circle": False,
                "line": False,
                "point": False,
                "linestrip": False,
                "ai_polygon": False,
                "ai_mask": False,
            },
        )
        super().__init__(*args, **kwargs)

        self.camera_cells: list[CameraCell] = []
        self.grid_cols = 2
        self.grid_rows = 2
        self.cell_padding = 5
        self.scale = 1.0
        
        # Per-cell zoom (scale) for detailed editing
        self.cell_scales: dict[int, float] = {}  # cell_index -> scale
        self.cell_scroll_offsets: dict[int, QPointF] = {}  # cell_index -> (offset_x, offset_y)
        
        # Fullscreen mode: if not None, only show this camera cell fullscreen
        self.fullscreen_cell_index: int | None = None
        
        # Track double-click to prevent mousePressEvent from interfering
        self._is_double_click = False
        
        # Current editing state
        self.current_cell_index: int | None = None
        self.current: Shape | None = None
        self.line = Shape()
        self.prevPoint = QPointF()
        self.prevMovePoint = QPointF()
        self.selectedShapes: list[Shape] = []
        self.selectedShapesCopy: list[Shape] = []
        self.hShape: Shape | None = None
        self.hVertex: int | None = None
        self.hEdge: int | None = None
        self.movingShape = False
        self.movingVertex = False
        self._move_start_pos: QPointF | None = None
        self._move_start_cell_idx: int | None = None
        
        # Shape management
        self.shapesBackups: list[dict[int, list[Shape]]] = []
        self.visible = {}
        self.snapping = True
        self.hShapeIsSelected = False
        
        # UI state
        self._cursor = CURSOR_DEFAULT
        self._painter = QtGui.QPainter()
        
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.WheelFocus)
        
        # Temporary projection markers: dict[camera_id] -> (x, y) or None
        # Used to show where BEV mouse position projects to on each camera
        self._bev_projection_markers: dict[str, QPointF | None] = {}
        
        # Menus
        self.menus = (QtWidgets.QMenu(), QtWidgets.QMenu())

    def loadMultiCameraData(self, camera_data: list[dict]):
        """Load multi-camera data
        
        Args:
            camera_data: List of dicts with keys: camera_id, image_path, calibration_path (optional)
        """
        self.camera_cells = []
        # Reset zoom, scroll, and fullscreen when loading new data
        self.cell_scales = {}
        self.cell_scroll_offsets = {}
        self.fullscreen_cell_index = None
        
        for data in camera_data:
            camera_id = data["camera_id"]
            image_path = data["image_path"]
            calibration_path = data.get("calibration_path")
            
            # Load image
            image = QtGui.QImage(image_path)
            if image.isNull():
                logger.warning(f"Failed to load image: {image_path}")
                continue
            
            pixmap = QtGui.QPixmap.fromImage(image)
            
            # Create cell (will be positioned in paintEvent)
            cell = CameraCell(
                camera_id=camera_id,
                image_path=image_path,
                image=image,
                pixmap=pixmap,
                cell_rect=QtCore.QRectF(),
                image_rect=QtCore.QRectF(),
                shapes=[],
                calibration_path=calibration_path,
            )
            self.camera_cells.append(cell)
        
        # Calculate grid layout
        num_cameras = len(self.camera_cells)
        if num_cameras == 0:
            return
        
        self.grid_cols = math.ceil(math.sqrt(num_cameras))
        self.grid_rows = math.ceil(num_cameras / self.grid_cols)
        
        self.update()

    def _get_cell_at_position(self, pos: QPointF) -> tuple[int, CameraCell] | None:
        """Get the camera cell at the given position"""
        # Ensure layout is calculated
        if self.camera_cells and (not self.camera_cells[0].cell_rect.width() or not self.camera_cells[0].cell_rect.height()):
            self._calculate_cell_layout(self.width(), self.height())
        
        # In fullscreen mode, always return the fullscreen cell if position is within widget
        if self.fullscreen_cell_index is not None:
            idx = self.fullscreen_cell_index
            if 0 <= idx < len(self.camera_cells):
                cell = self.camera_cells[idx]
                if cell.image_rect.contains(pos):
                    return idx, cell
            return None
        
        # Multi-camera mode: check image_rect instead of cell_rect to only match clicks on actual image area
        for idx, cell in enumerate(self.camera_cells):
            if cell.image_rect.width() > 0 and cell.image_rect.height() > 0:
                if cell.image_rect.contains(pos):
                    return idx, cell
        return None

    def _global_to_cell_coords(self, global_pos: QPointF, cell: CameraCell) -> QPointF:
        """Convert global canvas coordinates to cell-local image coordinates"""
        # Get position relative to image rect (which is already in global coordinates)
        image_local = global_pos - cell.image_rect.topLeft()
        
        # Scale to original image coordinates (accounting for zoom)
        scale_x = cell.image.width() / cell.image_rect.width() if cell.image_rect.width() > 0 else 1.0
        scale_y = cell.image.height() / cell.image_rect.height() if cell.image_rect.height() > 0 else 1.0
        
        return QPointF(image_local.x() * scale_x, image_local.y() * scale_y)

    def _cell_to_global_coords(self, cell_pos: QPointF, cell: CameraCell) -> QPointF:
        """Convert cell-local image coordinates to global canvas coordinates"""
        # Scale to cell image size
        scale_x = cell.image_rect.width() / cell.image.width() if cell.image.width() > 0 else 1.0
        scale_y = cell.image_rect.height() / cell.image.height() if cell.image.height() > 0 else 1.0
        
        scaled_pos = QPointF(cell_pos.x() * scale_x, cell_pos.y() * scale_y)
        
        # Add cell and image offsets
        return scaled_pos + cell.image_rect.topLeft()

    def _is_point_in_cell_bounds(self, point: QPointF, cell: CameraCell) -> bool:
        """Check if a point (in cell coordinates) is within the cell image bounds"""
        return (0 <= point.x() <= cell.image.width() and 
                0 <= point.y() <= cell.image.height())

    def _is_shape_in_cell_bounds(self, shape: Shape, cell: CameraCell) -> bool:
        """Check if all points of a shape are within the cell bounds"""
        for point in shape.points:
            if not self._is_point_in_cell_bounds(point, cell):
                return False
        return True

    def _constrain_point_to_cell(self, point: QPointF, cell: CameraCell) -> QPointF:
        """Constrain a point to stay within cell bounds"""
        x = max(0, min(point.x(), cell.image.width()))
        y = max(0, min(point.y(), cell.image.height()))
        return QPointF(x, y)

    def _calculate_cell_layout(self, width: float, height: float):
        """Calculate cell positions and sizes"""
        if not self.camera_cells:
            return
        
        # Fullscreen mode: show only one camera fullscreen
        if self.fullscreen_cell_index is not None:
            idx = self.fullscreen_cell_index
            if 0 <= idx < len(self.camera_cells):
                cell = self.camera_cells[idx]
                
                # Use entire widget for fullscreen cell
                cell_rect = QtCore.QRectF(0, 0, width, height)
                
                # Calculate image rect to fill widget while maintaining aspect ratio
                img_aspect = cell.image.width() / cell.image.height() if cell.image.height() > 0 else 1.0
                widget_aspect = width / height if height > 0 else 1.0
                
                if img_aspect > widget_aspect:
                    # Image is wider, fit to width
                    img_width = width
                    img_height = width / img_aspect
                else:
                    # Image is taller, fit to height
                    img_height = height
                    img_width = height * img_aspect
                
                # Center image in widget
                img_x = (width - img_width) / 2
                img_y = (height - img_height) / 2
                
                image_rect = QtCore.QRectF(img_x, img_y, img_width, img_height)
                
                # Update cell with new rects
                self.camera_cells[idx] = cell._replace(
                    cell_rect=cell_rect,
                    image_rect=image_rect,
                )
                
                # Set other cells to empty rects (not visible)
                for other_idx, other_cell in enumerate(self.camera_cells):
                    if other_idx != idx:
                        self.camera_cells[other_idx] = other_cell._replace(
                            cell_rect=QtCore.QRectF(),
                            image_rect=QtCore.QRectF(),
                        )
            return
        
        # Multi-camera mode: calculate grid layout
        # Calculate cell size
        total_padding = self.cell_padding * (self.grid_cols + 1)
        cell_width = (width - total_padding) / self.grid_cols
        total_padding_v = self.cell_padding * (self.grid_rows + 1)
        cell_height = (height - total_padding_v) / self.grid_rows
        
        # Header height for camera label (reserved space at top of each cell)
        header_height = 25
        
        # Position each cell
        for idx, cell in enumerate(self.camera_cells):
            row = idx // self.grid_cols
            col = idx % self.grid_cols
            
            x = self.cell_padding + col * (cell_width + self.cell_padding)
            y = self.cell_padding + row * (cell_height + self.cell_padding)
            
            cell_rect = QtCore.QRectF(x, y, cell_width, cell_height)
            
            # Get cell zoom scale (default 1.0 if not set)
            cell_scale = self.cell_scales.get(idx, 1.0)
            
            # Calculate available space for image (cell height minus header)
            available_height = cell_height - header_height
            
            # Calculate base image rect within available space (maintain aspect ratio)
            img_aspect = cell.image.width() / cell.image.height() if cell.image.height() > 0 else 1.0
            cell_aspect = cell_width / available_height if available_height > 0 else 1.0
            
            if img_aspect > cell_aspect:
                # Image is wider, fit to width
                base_img_width = cell_width
                base_img_height = cell_width / img_aspect
            else:
                # Image is taller, fit to available height
                base_img_height = available_height
                base_img_width = available_height * img_aspect
            
            # Apply zoom scale
            img_width = base_img_width * cell_scale
            img_height = base_img_height * cell_scale
            
            # Get scroll offset for this cell
            scroll_offset = self.cell_scroll_offsets.get(idx, QPointF(0, 0))
            
            # Position image below header, then apply scroll offset
            img_x = x + (cell_width - img_width) / 2 + scroll_offset.x()
            img_y = y + header_height + (available_height - img_height) / 2 + scroll_offset.y()
            
            image_rect = QtCore.QRectF(img_x, img_y, img_width, img_height)
            
            # Update cell with new rects
            self.camera_cells[idx] = cell._replace(
                cell_rect=cell_rect,
                image_rect=image_rect,
            )
    
    def setBEVProjectionMarkers(self, markers: dict[str, QPointF | None]):
        """
        Set temporary projection markers showing where BEV mouse position projects to on each camera.
        
        Args:
            markers: dict mapping camera_id to (x, y) pixel coordinates or None
        """
        self._bev_projection_markers = markers.copy()
        self.update()

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        if not self.camera_cells:
            return super().paintEvent(a0)
        
        # Calculate layout
        self._calculate_cell_layout(self.width(), self.height())
        
        p = self._painter
        p.begin(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        
        # Draw each camera cell
        for idx, cell in enumerate(self.camera_cells):
            # Skip cells with empty rects (not visible)
            if cell.image_rect.width() == 0 or cell.image_rect.height() == 0:
                continue
            
            # Draw cell background (only in multi-camera mode, not fullscreen)
            if self.fullscreen_cell_index is None:
                p.setPen(QtGui.QPen(QtGui.QColor(100, 100, 100), 2))
                p.setBrush(QtGui.QBrush(QtGui.QColor(50, 50, 50)))
                p.drawRect(cell.cell_rect)
            
            # Draw header bar for camera label (separate from image, no overlap)
            header_height = 25
            # Header is always at top of cell
            header_rect = QtCore.QRectF(
                cell.cell_rect.x(),
                cell.cell_rect.y(),
                cell.cell_rect.width(),
                header_height
            )
            
            # Draw header background
            header_bg_color = QtGui.QColor(60, 60, 60, 255)  # Dark gray, fully opaque
            p.fillRect(header_rect, header_bg_color)
            
            # Draw header border (bottom border only to separate from image)
            p.setPen(QtGui.QPen(QtGui.QColor(100, 100, 100), 1))
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawLine(
                int(header_rect.left()),
                int(header_rect.bottom()),
                int(header_rect.right()),
                int(header_rect.bottom())
            )
            
            # Draw camera label in header
            font = p.font()
            font.setBold(True)
            font.setPointSize(11)
            p.setFont(font)
            
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
            metrics = p.fontMetrics()
            text_height = metrics.height()
            label_x = header_rect.x() + 8
            label_y = header_rect.y() + (header_height - text_height) / 2 + metrics.ascent()
            p.drawText(int(label_x), int(label_y), cell.camera_id)
            
            # Draw image (already positioned below header in layout calculation)
            p.drawPixmap(cell.image_rect.toRect(), cell.pixmap)
            
            # Draw border around image (only in multi-camera mode)
            if self.fullscreen_cell_index is None:
                p.setPen(QtGui.QPen(QtGui.QColor(200, 200, 200), 1))
                p.setBrush(QtCore.Qt.NoBrush)
                p.drawRect(cell.image_rect)
            
            # Draw shapes for this cell
            # Note: Set Shape.scale = 1.0 because we already transform points to global coordinates
            # Shape.scale is used in Shape._scale_point() to scale points, but we've already done the transformation
            Shape.scale = 1.0
            for shape in cell.shapes:
                if self.isVisible(shape):
                    # Set fill based on selection or BEV highlight
                    bev_highlighted = getattr(shape, '_bev_highlighted', False)
                    shape.fill = shape.selected or shape == self.hShape or bev_highlighted
                    
                    # If BEV highlighted, use a different color
                    original_line_color = None
                    if bev_highlighted:
                        original_line_color = shape.line_color
                        shape.line_color = QtGui.QColor(255, 255, 0, 255)  # Yellow highlight
                    
                    # Transform shape points to global coordinates for drawing
                    original_points = shape.points.copy()
                    for i, point in enumerate(shape.points):
                        global_point = self._cell_to_global_coords(point, cell)
                        shape.points[i] = global_point
                    
                    shape.paint(p)
                    
                    # Restore original points and color
                    shape.points = original_points
                    if original_line_color:
                        shape.line_color = original_line_color
            
            # Draw selectedShapesCopy if any (for copy/move preview)
            # Set Shape.scale = 1.0 because we already transform points to global coordinates
            Shape.scale = 1.0
            if self.selectedShapesCopy and self._move_start_cell_idx == idx:
                for shape_copy in self.selectedShapesCopy:
                    # Transform copy points to global coordinates
                    original_points = shape_copy.points.copy()
                    for i, point in enumerate(shape_copy.points):
                        global_point = self._cell_to_global_coords(point, cell)
                        shape_copy.points[i] = global_point
                    
                    shape_copy.paint(p)
                    
                    # Restore original points
                    shape_copy.points = original_points
            
            # Draw BEV projection markers (temporary markers showing where BEV mouse projects to)
            if cell.camera_id in self._bev_projection_markers:
                marker_pos = self._bev_projection_markers[cell.camera_id]
                if marker_pos is not None and cell.image.width() > 0 and cell.image.height() > 0:
                    # Check if marker is within image bounds (in cell/image coordinates)
                    if (0 <= marker_pos.x() < cell.image.width() and 
                        0 <= marker_pos.y() < cell.image.height()):
                        # Convert marker position (in cell/image coordinates) to global coordinates
                        global_marker = self._cell_to_global_coords(marker_pos, cell)
                        
                        # Double-check if marker is within image rect bounds
                        if cell.image_rect.contains(global_marker):
                            # Draw a crosshair/circle marker
                            marker_size = 8
                            p.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0), 1))  # Magenta color
                            p.setBrush(QtCore.Qt.NoBrush)
                            
                            # Draw crosshair
                            p.drawLine(
                                int(global_marker.x() - marker_size),
                                int(global_marker.y()),
                                int(global_marker.x() + marker_size),
                                int(global_marker.y())
                            )
                            p.drawLine(
                                int(global_marker.x()),
                                int(global_marker.y() - marker_size),
                                int(global_marker.x()),
                                int(global_marker.y() + marker_size)
                            )
                            
                            # Draw circle around marker
                            p.drawEllipse(
                                int(global_marker.x() - marker_size),
                                int(global_marker.y() - marker_size),
                                marker_size * 2,
                                marker_size * 2
                            )
        
        # Draw crosshair when drawing (like default Canvas)
        # Crosshair should be drawn after images but before shapes (like Canvas default)
        if (
            self._crosshair.get(self._createMode, False)
            and self.drawing()
            and self.prevMovePoint is not None
            and self.current_cell_index is not None
        ):
            cell = self.camera_cells[self.current_cell_index]
            # Check if point is within cell bounds (like outOfPixmap check in Canvas)
            if self._is_point_in_cell_bounds(self.prevMovePoint, cell):
                # Convert cell coordinates to global for crosshair
                crosshair_pos = self._cell_to_global_coords(self.prevMovePoint, cell)
                
                # Draw crosshair lines across the entire widget (like Canvas default)
                p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 1))
                p.drawLine(
                    0,
                    int(crosshair_pos.y()),
                    self.width() - 1,
                    int(crosshair_pos.y()),
                )
                p.drawLine(
                    int(crosshair_pos.x()),
                    0,
                    int(crosshair_pos.x()),
                    self.height() - 1,
                )
        
        # Draw current drawing shape and line preview (like Canvas default)
        # These should be drawn after all other shapes
        # Set Shape.scale = 1.0 because we already transform points to global coordinates
        Shape.scale = 1.0
        if self.current and self.current_cell_index is not None:
            cell = self.camera_cells[self.current_cell_index]
            original_points = self.current.points.copy()
            for i, point in enumerate(self.current.points):
                global_point = self._cell_to_global_coords(point, cell)
                self.current.points[i] = global_point
            self.current.paint(p)
            self.current.points = original_points
            
            # Draw line preview (for rectangle, circle, etc.)
            if self.line.points:
                original_line_points = self.line.points.copy()
                for i, point in enumerate(self.line.points):
                    global_point = self._cell_to_global_coords(point, cell)
                    self.line.points[i] = global_point
                self.line.paint(p)
                self.line.points = original_line_points
        
        # Draw preview shape with fill (like default Canvas) for polygon mode
        # This should be drawn last, only for polygon mode with fillDrawing enabled
        # Shape.scale is already set to 1.0 above
        if (
            self.current
            and self.current_cell_index is not None
            and self._createMode == "polygon"
            and self.fillDrawing()
            and len(self.current.points) >= 2
            and len(self.line.points) >= 2
        ):
            cell = self.camera_cells[self.current_cell_index]
            # Create preview shape with fill (like Canvas default)
            drawing_shape = self.current.copy()
            # Add the last point from line to complete the preview (self.line[1] in Canvas)
            if len(self.line.points) >= 2:
                last_line_point = self.line.points[1]  # Use index 1 like Canvas default
                drawing_shape.addPoint(last_line_point)
            
            # Ensure fill_color is set and not transparent
            if drawing_shape.fill_color is not None:
                if drawing_shape.fill_color.getRgb()[3] == 0:
                    # Force to be opaque if transparent
                    drawing_shape.fill_color.setAlpha(64)
            
            # Set fill properties
            drawing_shape.fill = True
            drawing_shape.selected = True
            
            # Transform points to global coordinates
            original_preview_points = drawing_shape.points.copy()
            for i, point in enumerate(drawing_shape.points):
                global_point = self._cell_to_global_coords(point, cell)
                drawing_shape.points[i] = global_point
            
            drawing_shape.paint(p)
            drawing_shape.points = original_preview_points
        
        p.end()

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        self.update()
        super().resizeEvent(a0)

    def enterEvent(self, a0: QtCore.QEvent) -> None:
        """Handle mouse enter event"""
        self.overrideCursor(self._cursor)
        self._update_status()

    def leaveEvent(self, a0: QtCore.QEvent) -> None:
        """Handle mouse leave event - restore cursor and clear highlight"""
        self.unHighlight()
        # Ensure cursor is restored
        self._cursor = CURSOR_DEFAULT
        self.setCursor(CURSOR_DEFAULT)
        self._update_status()

    def focusOutEvent(self, a0: QtGui.QFocusEvent) -> None:
        """Handle focus out event - restore cursor"""
        # Always restore cursor when losing focus
        self._cursor = CURSOR_DEFAULT
        self.setCursor(CURSOR_DEFAULT)
        self._update_status()

    def mouseMoveEvent(self, a0: QtGui.QMouseEvent) -> None:
        pos = a0.localPos()
        
        # Ensure layout is calculated
        if self.camera_cells and (not self.camera_cells[0].cell_rect.width() or not self.camera_cells[0].cell_rect.height()):
            self._calculate_cell_layout(self.width(), self.height())
        
        # Find which cell the mouse is over
        cell_info = self._get_cell_at_position(pos)
        
        if cell_info is None:
            # Mouse is outside any cell - restore cursor and clear highlight
            self._cursor = CURSOR_DEFAULT
            self.setCursor(CURSOR_DEFAULT)
            self.unHighlight()
            return
        
        cell_idx, cell = cell_info
        
        # Convert to cell coordinates
        cell_pos = self._global_to_cell_coords(pos, cell)
        
        # Constrain to cell bounds
        cell_pos = self._constrain_point_to_cell(cell_pos, cell)
        
        self.mouseMoved.emit(cell_pos)
        self.prevMovePoint = cell_pos
        
        if self.drawing():
            if self.current_cell_index != cell_idx:
                # Started drawing in different cell, cancel
                self.current = None
                self.current_cell_index = None
                return
            
            if not self.current:
                self.repaint()
                return
            
            # Update drawing
            if not self._is_point_in_cell_bounds(cell_pos, cell):
                cell_pos = self._constrain_point_to_cell(cell_pos, cell)
            
            # Set line shape_type (like Canvas default) so it draws correctly
            if self._createMode in ["ai_polygon", "ai_mask"]:
                self.line.shape_type = "points"
            else:
                self.line.shape_type = self._createMode
            
            # Update line based on create mode
            if self._createMode == "rectangle":
                self.line.points = [self.current[0], cell_pos]
                self.line.point_labels = [1, 1]
                self.line.close()
            elif self._createMode == "circle":
                self.line.points = [self.current[0], cell_pos]
                self.line.point_labels = [1, 1]
            elif self._createMode in ["polygon", "linestrip"]:
                self.line.points = [self.current[-1], cell_pos]
                self.line.point_labels = [1, 1]
            
            self.repaint()
        else:
            # Editing mode
            if Qt.LeftButton & a0.buttons():
                # Handle dragging for moving shapes/vertices
                if self.movingVertex and self.hShape and self.hVertex is not None:
                    # Check if shape is locked - locked shapes cannot be edited
                    is_locked = getattr(self.hShape, '_bev_locked', False)
                    if is_locked:
                        # Stop moving if shape is locked
                        self.movingVertex = False
                        self.hShape = None
                        self.hVertex = None
                        return
                    
                    # Moving vertex
                    new_pos = cell_pos
                    new_pos = self._constrain_point_to_cell(new_pos, cell)
                    self.hShape.points[self.hVertex] = new_pos
                    # Ensure shape stays in bounds
                    if not self._is_shape_in_cell_bounds(self.hShape, cell):
                        # Constrain all points
                        for i, point in enumerate(self.hShape.points):
                            self.hShape.points[i] = self._constrain_point_to_cell(point, cell)
                    
                    # Unlink from BEV if shape is edited manually
                    if self.hShape.other_data.get("from_bev", False):
                        self.hShape.other_data["from_bev"] = False
                    
                    self.shapeMoved.emit()
                    self.repaint()
                elif self.movingShape and self.selectedShapes and self._move_start_pos is not None and self._move_start_cell_idx is not None:
                    # Check if any selected shape is locked - locked shapes cannot be moved
                    has_locked = any(getattr(shape, '_bev_locked', False) for shape in self.selectedShapes)
                    if has_locked:
                        # Stop moving if any shape is locked
                        self.movingShape = False
                        self.selectedShapes = []
                        self.selectedShapesCopy = []
                        self._move_start_pos = None
                        self._move_start_cell_idx = None
                        return
                    
                    # Moving selected shapes
                    if self._move_start_cell_idx == cell_idx:
                        # Calculate offset
                        offset = cell_pos - self._move_start_pos
                        # Move all selected shapes
                        for shape in self.selectedShapes:
                            # Find which cell this shape belongs to
                            for idx, c in enumerate(self.camera_cells):
                                if shape in c.shapes:
                                    # Move shape
                                    for i, point in enumerate(shape.points):
                                        new_point = point + offset
                                        new_point = self._constrain_point_to_cell(new_point, c)
                                        shape.points[i] = new_point
                                    
                                    # Ensure shape stays in bounds
                                    if not self._is_shape_in_cell_bounds(shape, c):
                                        # Constrain all points
                                        for i, point in enumerate(shape.points):
                                            shape.points[i] = self._constrain_point_to_cell(point, c)
                                    
                                    # Unlink from BEV if shape is moved manually
                                    if shape.other_data.get("from_bev", False):
                                        shape.other_data["from_bev"] = False
                                    break
                        self._move_start_pos = cell_pos
                        self.shapeMoved.emit()
                        self.repaint()
                elif self.selectedShapesCopy and a0.buttons() & Qt.RightButton and self._move_start_cell_idx is not None:
                    # Moving copy (right-click drag) - only if in same cell
                    if self._move_start_cell_idx == cell_idx:
                        # Calculate offset for copy
                        if self._move_start_pos is not None:
                            offset = cell_pos - self._move_start_pos
                            # Update copy positions
                            for shape_copy in self.selectedShapesCopy:
                                for i, point in enumerate(shape_copy.points):
                                    new_point = point + offset
                                    new_point = self._constrain_point_to_cell(new_point, cell)
                                    shape_copy.points[i] = new_point
                            self._move_start_pos = cell_pos
                            self.repaint()
                else:
                    # Just hovering - highlight shapes
                    # Reset moving flags if mouse button is released but we're still in moving state
                    if self.movingShape or self.movingVertex:
                        self.movingShape = False
                        self.movingVertex = False
                        self._move_start_pos = None
                        self._move_start_cell_idx = None
                    self._update_hover_shape(cell_pos, cell)
            else:
                # Just hovering - highlight shapes
                # Reset moving flags if mouse button is released but we're still in moving state
                if self.movingShape or self.movingVertex:
                    self.movingShape = False
                    self.movingVertex = False
                    self._move_start_pos = None
                    self._move_start_cell_idx = None
                self._update_hover_shape(cell_pos, cell)
        
        self._update_status()

    def mousePressEvent(self, a0: QtGui.QMouseEvent) -> None:
        pos = a0.localPos()
        
        # Ensure layout is calculated
        if self.camera_cells and (not self.camera_cells[0].cell_rect.width() or not self.camera_cells[0].cell_rect.height()):
            self._calculate_cell_layout(self.width(), self.height())
        
        cell_info = self._get_cell_at_position(pos)
        
        if cell_info is None:
            # Reset double-click flag if clicking outside
            self._is_double_click = False
            
            # ALWAYS reset moving flags when clicking outside - this is critical
            self.movingShape = False
            self.movingVertex = False
            self._move_start_pos = None
            self._move_start_cell_idx = None
            self.selectedShapesCopy = []
            
            # Clear hover state when clicking outside
            self.unHighlight()
            
            # Deselect all shapes when clicking outside (unless Ctrl is held)
            if not (a0.modifiers() & Qt.ControlModifier):
                # Clear selection flags on all shapes across all cells
                for cell in self.camera_cells:
                    for s in cell.shapes:
                        s.selected = False
                self.selectedShapes = []
                self.selectionChanged.emit([])
            
            self.repaint()
            return
        
        cell_idx, cell = cell_info
        cell_pos = self._global_to_cell_coords(pos, cell)
        cell_pos = self._constrain_point_to_cell(cell_pos, cell)
        
        # If this is part of a double-click, skip handling click
        if self._is_double_click:
            # Reset flag after double-click handling
            self._is_double_click = False
            # Still update hover state
            self._update_hover_shape(cell_pos, cell)
            self.prevPoint = cell_pos
            return
        
        if a0.button() == Qt.LeftButton:
            if self.drawing():
                if self.current_cell_index is None:
                    self.current_cell_index = cell_idx
                    self.current = Shape(
                        label="",
                        shape_type=self._createMode,
                    )
                    self.current.addPoint(cell_pos)
                elif self.current_cell_index == cell_idx:
                    # Ensure point is within bounds
                    cell_pos = self._constrain_point_to_cell(cell_pos, cell)
                    self.current.addPoint(cell_pos)
                    
                    # Check if shape should be closed
                    if self._createMode in ["rectangle", "circle", "line", "point"]:
                        if len(self.current.points) >= 2:
                            # Finalize shape
                            if self._createMode == "rectangle" and len(self.current.points) == 2:
                                # Ensure rectangle stays in bounds
                                p1, p2 = self.current.points[0], self.current.points[1]
                                p1 = self._constrain_point_to_cell(p1, cell)
                                p2 = self._constrain_point_to_cell(p2, cell)
                                self.current.points = [p1, p2]
                            self.newShape.emit()
                    elif self._createMode == "polygon":
                        if len(self.current.points) >= 3 and self.closeEnough(cell_pos, self.current[0]):
                            self.current.close()
                            self.newShape.emit()
            else:
                # Editing mode - update hover first, then handle click
                # Update hover state before handling click
                self._update_hover_shape(cell_pos, cell)
                self._handle_edit_click(cell_pos, cell, a0)
        
        self.prevPoint = cell_pos
        self.repaint()

    def mouseReleaseEvent(self, a0: QtGui.QMouseEvent) -> None:
        """Handle mouse release"""
        # Always reset moving flags on mouse release, regardless of button or position
        if self.movingShape or self.movingVertex:
            # End moving
            if self.movingShape and self.selectedShapes:
                # Store shapes after move
                self.storeShapes()
                self.shapeMoved.emit()
            elif self.movingVertex and self.hShape:
                # Store shapes after vertex move
                self.storeShapes()
                self.shapeMoved.emit()
            
            self.movingShape = False
            self.movingVertex = False
            self._move_start_pos = None
            self._move_start_cell_idx = None
        
        if a0.button() == Qt.RightButton:
            # Handle right-click release (for copy mode)
            if self.selectedShapesCopy:
                # End copy mode - shapes will be finalized by menu action
                pass
        
        # Update hover state after release
        pos = a0.localPos()
        cell_info = self._get_cell_at_position(pos)
        if cell_info is not None:
            cell_idx, cell = cell_info
            cell_pos = self._global_to_cell_coords(pos, cell)
            cell_pos = self._constrain_point_to_cell(cell_pos, cell)
            self._update_hover_shape(cell_pos, cell)
        else:
            # Mouse released outside any cell - clear hover
            self.unHighlight()
        
        self.repaint()
        self._update_status()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        """Handle mouse wheel for zooming into individual camera cells"""
        pos = event.position() if hasattr(event, 'position') else event.pos()
        pos_qpoint = QPointF(pos)
        
        # Find which cell the mouse is over
        cell_info = self._get_cell_at_position(pos_qpoint)
        
        if cell_info is None:
            return
        
        cell_idx, cell = cell_info
        
        # Get current zoom and scroll for this cell
        current_scale = self.cell_scales.get(cell_idx, 1.0)
        current_scroll = self.cell_scroll_offsets.get(cell_idx, QPointF(0, 0))
        
        # Calculate zoom factor (scroll up = zoom in, scroll down = zoom out)
        zoom_factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        new_scale = current_scale * zoom_factor
        
        # Limit zoom range (0.1x to 10x)
        new_scale = max(0.1, min(10.0, new_scale))
        
        if abs(new_scale - current_scale) < 0.01:
            return
        
        # Get mouse position relative to cell image rect (before zoom)
        mouse_rel_to_image = pos_qpoint - cell.image_rect.topLeft()
        
        # Get the corresponding point in original image coordinates
        img_scale_x_before = cell.image.width() / cell.image_rect.width() if cell.image_rect.width() > 0 else 1.0
        img_scale_y_before = cell.image.height() / cell.image_rect.height() if cell.image_rect.height() > 0 else 1.0
        img_point = QPointF(
            mouse_rel_to_image.x() * img_scale_x_before,
            mouse_rel_to_image.y() * img_scale_y_before
        )
        
        # Update zoom
        self.cell_scales[cell_idx] = new_scale
        
        # Recalculate layout to get new image rect
        self._calculate_cell_layout(self.width(), self.height())
        cell = self.camera_cells[cell_idx]
        
        # Calculate where the image point should be in screen space with new zoom
        img_scale_x_after = cell.image_rect.width() / cell.image.width() if cell.image.width() > 0 else 1.0
        img_scale_y_after = cell.image_rect.height() / cell.image.height() if cell.image.height() > 0 else 1.0
        target_screen_pos = QPointF(
            cell.image_rect.left() + img_point.x() * img_scale_x_after,
            cell.image_rect.top() + img_point.y() * img_scale_y_after
        )
        
        # Adjust scroll offset so the image point stays under the mouse
        scroll_delta = pos_qpoint - target_screen_pos
        new_scroll = current_scroll + scroll_delta
        
        # Restore cursor after zoom to prevent stuck cursor
        self.restoreCursor()
        self.cell_scroll_offsets[cell_idx] = new_scroll
        
        # Recalculate layout again with new scroll offset
        self._calculate_cell_layout(self.width(), self.height())
        
        # Update hover state after zoom to ensure hover detection works correctly
        # Re-check hover at current mouse position
        cell = self.camera_cells[cell_idx]
        cell_pos = self._global_to_cell_coords(pos_qpoint, cell)
        cell_pos = self._constrain_point_to_cell(cell_pos, cell)
        self._update_hover_shape(cell_pos, cell)
        
        self.update()

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle double-click to toggle fullscreen mode for a cell"""
        if event.button() != Qt.LeftButton:
            return
        
        # Mark as double-click to prevent mousePressEvent from interfering
        self._is_double_click = True
        
        # Cancel any ongoing drag/move operations
        if self.movingShape:
            self.movingShape = False
            self._move_start_pos = None
            self._move_start_cell_idx = None
        if self.movingVertex:
            self.movingVertex = False
            self._move_start_pos = None
            self._move_start_cell_idx = None
        
        pos = event.localPos()
        cell_info = self._get_cell_at_position(pos)
        
        if cell_info is None:
            # If clicking outside cells in fullscreen mode, exit fullscreen
            if self.fullscreen_cell_index is not None:
                self.fullscreen_cell_index = None
                self._calculate_cell_layout(self.width(), self.height())
                # Update hover state after exiting fullscreen
                self.unHighlight()
                self.update()
            return
        
        cell_idx, cell = cell_info
        
        # Toggle fullscreen mode
        if self.fullscreen_cell_index == cell_idx:
            # Already in fullscreen for this cell, exit fullscreen
            self.fullscreen_cell_index = None
        else:
            # Enter fullscreen for this cell
            self.fullscreen_cell_index = cell_idx
            # Reset zoom and scroll when entering fullscreen
            if cell_idx in self.cell_scales:
                del self.cell_scales[cell_idx]
            if cell_idx in self.cell_scroll_offsets:
                del self.cell_scroll_offsets[cell_idx]
        
        # Recalculate layout
        self._calculate_cell_layout(self.width(), self.height())
        
        # Update hover state after layout change to ensure shapes can be highlighted
        # Get updated cell after layout recalculation
        cell = self.camera_cells[cell_idx]
        cell_pos = self._global_to_cell_coords(pos, cell)
        cell_pos = self._constrain_point_to_cell(cell_pos, cell)
        
        # Set current_cell_index so editing works immediately after zoom to fullscreen
        self.current_cell_index = cell_idx
        
        # Set prevPoint and prevMovePoint so mouse move events work correctly
        self.prevPoint = cell_pos
        self.prevMovePoint = cell_pos
        
        # Clear previous hover state first and restore cursor
        self.unHighlight()
        
        # Update hover state with current mouse position
        self._update_hover_shape(cell_pos, cell)
        
        # Reset double-click flag immediately so next click/drag works right away
        # This allows immediate editing after zoom to fullscreen
        self._is_double_click = False
        
        # If there's a hovered shape and mouse button is still pressed (user is dragging),
        # set up moving state immediately so box can be dragged right away
        if self.hShape and self.mode == CanvasMode.EDIT:
            # Check if shape is locked - locked shapes cannot be edited
            is_locked = getattr(self.hShape, '_bev_locked', False)
            if not is_locked:
                # Pre-select the shape so it can be dragged immediately
                if self.hShape not in self.selectedShapes:
                    # Clear previous selections
                    for s in self.selectedShapes:
                        s.selected = False
                    self.selectedShapes = [self.hShape]
                    self.hShape.selected = True
                    self.selectionChanged.emit(self.selectedShapes)
                
                # If mouse button is still pressed (user is dragging after double-click),
                # set up moving state immediately
                if event.buttons() & Qt.LeftButton:
                    if self.hVertex is not None:
                        # Start moving vertex
                        self.movingVertex = True
                        self.movingShape = False
                        self._move_start_pos = cell_pos
                        self._move_start_cell_idx = cell_idx
                    elif self.hShape.containsPoint(cell_pos):
                        # Start moving shape
                        self.movingShape = True
                        self.movingVertex = False
                        self._move_start_pos = cell_pos
                        self._move_start_cell_idx = cell_idx
                        self.storeShapes()
        
        self.update()

    def _update_hover_shape(self, pos: QPointF, cell: CameraCell):
        """Update highlighted shape based on mouse position"""
        # Set Shape.scale = 1.0 because shapes are stored in cell coordinates
        # and we're checking against cell coordinates
        Shape.scale = 1.0
        
        # Check shapes in reverse order (top to bottom)
        for shape in reversed(cell.shapes):
            if not self.isVisible(shape):
                continue
            
            # Check vertex
            index = shape.nearestVertex(pos, self.epsilon)
            if index is not None:
                self.hShape = shape
                self.hVertex = index
                self.hEdge = None
                shape.highlightVertex(index, Shape.MOVE_VERTEX)
                self.overrideCursor(CURSOR_POINT)
                self.repaint()
                return
            
            # Check if inside shape
            if shape.containsPoint(pos):
                self.hShape = shape
                self.hVertex = None
                self.hEdge = None
                self.overrideCursor(CURSOR_GRAB)
                self.repaint()
                return
        
        # Nothing found - restore cursor and clear highlight
        if self.hShape:
            self.hShape.highlightClear()
        self.hShape = None
        self.hVertex = None
        self.hEdge = None
        # Always restore cursor when not hovering over any shape
        self._cursor = CURSOR_DEFAULT
        self.setCursor(CURSOR_DEFAULT)
        self.repaint()

    def _handle_edit_click(self, pos: QPointF, cell: CameraCell, event: QtGui.QMouseEvent):
        """Handle click in edit mode"""
        # Set Shape.scale = 1.0 because shapes are stored in cell coordinates
        Shape.scale = 1.0
        
        # IMPORTANT: Reset moving flags before handling click to ensure clean state
        # This prevents stuck state if previous move wasn't properly ended
        if not (self.movingShape or self.movingVertex):
            # Only reset if we're not currently moving (to allow continuing a move)
            pass
        else:
            # If flags are still set, something went wrong - reset them
            self.movingShape = False
            self.movingVertex = False
            self._move_start_pos = None
            self._move_start_cell_idx = None
        
        # Check for right-click copy mode
        is_right_click = event.button() == Qt.RightButton
        is_control = event.modifiers() & Qt.ControlModifier
        
        if self.hShape:
            # Check if shape is locked - locked shapes cannot be edited
            is_locked = getattr(self.hShape, '_bev_locked', False)
            
            if is_locked:
                # Shape is locked, don't allow editing
                return
            
            if self.hVertex is not None:
                # Start moving vertex
                self.movingVertex = True
                self.movingShape = False
                self._move_start_pos = pos
                # Find which cell this shape belongs to
                for idx, c in enumerate(self.camera_cells):
                    if self.hShape in c.shapes:
                        self._move_start_cell_idx = idx
                        break
            elif self.hShape.containsPoint(pos):
                # Start moving shape
                if not is_control:
                    # Always select the shape when clicking on it (unless Ctrl is held)
                    # This ensures the shape can be moved even after clicking outside
                    if self.hShape not in self.selectedShapes:
                        # Clear previous selections and select this shape
                        for s in self.selectedShapes:
                            s.selected = False
                        self.selectedShapes = [self.hShape]
                        self.hShape.selected = True
                        self.selectionChanged.emit(self.selectedShapes)
                    elif not self.hShape.selected:
                        # Shape is in list but not marked as selected - fix it
                        self.hShape.selected = True
                else:
                    # Toggle selection with Ctrl
                    if self.hShape in self.selectedShapes:
                        self.selectedShapes.remove(self.hShape)
                        self.hShape.selected = False
                    else:
                        self.selectedShapes.append(self.hShape)
                        self.hShape.selected = True
                    self.selectionChanged.emit(self.selectedShapes)
                
                # IMPORTANT: Only start moving if shape is actually selected
                if self.hShape.selected and self.hShape in self.selectedShapes:
                    if is_right_click:
                        # Start copy mode - create copies
                        self.selectedShapesCopy = [s.copy() for s in self.selectedShapes]
                        self._move_start_pos = pos
                        # Find which cell this shape belongs to
                        for idx, c in enumerate(self.camera_cells):
                            if self.hShape in c.shapes:
                                self._move_start_cell_idx = idx
                                break
                    else:
                        # Start move mode
                        self.movingShape = True
                        self.movingVertex = False
                        self._move_start_pos = pos
                        # Find which cell this shape belongs to
                        for idx, c in enumerate(self.camera_cells):
                            if self.hShape in c.shapes:
                                self._move_start_cell_idx = idx
                                break
                        self.storeShapes()
            else:
                # Click outside shape - deselect unless Ctrl is held
                if not is_control:
                    # Clear selection flags on all shapes in this cell
                    for s in cell.shapes:
                        s.selected = False
                    self.selectedShapes = []
                    self.selectionChanged.emit([])
        else:
            # Click on empty space - deselect all unless Ctrl is held
            if not is_control:
                # Clear selection flags on all shapes in this cell
                for s in cell.shapes:
                    s.selected = False
                self.selectedShapes = []
                self.selectionChanged.emit([])

    def closeEnough(self, p1: QPointF, p2: QPointF) -> bool:
        """Check if two points are close enough"""
        return (p1 - p2).manhattanLength() < self.epsilon

    def drawing(self):
        return self.mode == CanvasMode.CREATE

    def editing(self):
        return self.mode == CanvasMode.EDIT

    def setEditing(self, value=True):
        self.mode = CanvasMode.EDIT if value else CanvasMode.CREATE

    @property
    def createMode(self):
        return self._createMode

    @createMode.setter
    def createMode(self, value):
        if value not in [
            "polygon",
            "rectangle",
            "circle",
            "line",
            "point",
            "linestrip",
            "ai_polygon",
            "ai_mask",
        ]:
            raise ValueError(f"Unsupported createMode: {value}")
        self._createMode = value

    def isVisible(self, shape):
        return self.visible.get(shape, True)

    def setShapeVisible(self, shape, visible):
        self.visible[shape] = visible
        self.update()

    def loadShapes(self, shapes: list[Shape], replace: bool = True):
        """Load shapes into the correct cells based on which cell each shape belongs to"""
        if not self.camera_cells:
            return
        
        # If current_cell_index is set and we're loading a small number of shapes,
        # assume they belong to the current cell (for normal drawing operations)
        # But if we're loading many shapes (like from undo), distribute them to correct cells
        if self.current_cell_index is not None and self.current_cell_index < len(self.camera_cells) and len(shapes) <= 10:
            # Small number of shapes - load into current cell (normal drawing)
            cell = self.camera_cells[self.current_cell_index]
            if replace:
                new_shapes = []
            else:
                new_shapes = cell.shapes.copy()
            
            # Ensure all shapes are within cell bounds
            for shape in shapes:
                if self._is_shape_in_cell_bounds(shape, cell):
                    new_shapes.append(shape)
                else:
                    logger.warning(f"Shape {shape.label} is outside cell bounds, skipping")
            
            # Update cell
            self.camera_cells[self.current_cell_index] = cell._replace(shapes=new_shapes)
        else:
            # Large number of shapes or current_cell_index not set - distribute to correct cells
            # Save fullscreen state and temporarily exit fullscreen to calculate layout correctly
            saved_fullscreen = self.fullscreen_cell_index
            if saved_fullscreen is not None:
                self.fullscreen_cell_index = None
            
            # Calculate layout in multi-camera mode so all cells have proper bounds
            if self.camera_cells and (not self.camera_cells[0].cell_rect.width() or not self.camera_cells[0].cell_rect.height()):
                self._calculate_cell_layout(self.width(), self.height())
            
            # Group shapes by which cell they belong to
            # First, check if shapes already exist in cells (by object identity)
            shapes_by_cell: dict[int, list[Shape]] = {}
            existing_shapes_by_cell: dict[int, set[Shape]] = {}
            for idx, cell in enumerate(self.camera_cells):
                existing_shapes_by_cell[idx] = set(cell.shapes)
            
            for shape in shapes:
                shape_assigned = False
                
                # First, check if shape already exists in a cell (by object identity)
                for idx, existing_shapes in existing_shapes_by_cell.items():
                    if shape in existing_shapes:
                        # Shape already exists in this cell, keep it here
                        if idx not in shapes_by_cell:
                            shapes_by_cell[idx] = []
                        shapes_by_cell[idx].append(shape)
                        shape_assigned = True
                        break
                
                if not shape_assigned:
                    # Shape doesn't exist yet, find which cell it belongs to by checking bounds
                    for idx, cell in enumerate(self.camera_cells):
                        if self._is_shape_in_cell_bounds(shape, cell):
                            if idx not in shapes_by_cell:
                                shapes_by_cell[idx] = []
                            shapes_by_cell[idx].append(shape)
                            shape_assigned = True
                            break
                
                if not shape_assigned:
                    logger.warning(f"Shape {shape.label} does not belong to any cell, skipping")
            
            # Load shapes into their respective cells
            for cell_idx, cell_shapes in shapes_by_cell.items():
                if cell_idx >= len(self.camera_cells):
                    continue
                
                cell = self.camera_cells[cell_idx]
                if replace:
                    new_shapes = []
                else:
                    new_shapes = cell.shapes.copy()
                
                new_shapes.extend(cell_shapes)
                self.camera_cells[cell_idx] = cell._replace(shapes=new_shapes)
            
            # Restore fullscreen mode if it was active
            if saved_fullscreen is not None:
                self.fullscreen_cell_index = saved_fullscreen
                self._calculate_cell_layout(self.width(), self.height())
        
        self.update()

    def getShapes(self) -> list[Shape]:
        """Get all shapes from all cells"""
        all_shapes = []
        for cell in self.camera_cells:
            all_shapes.extend(cell.shapes)
        return all_shapes

    def getShapesForCell(self, cell_idx: int) -> list[Shape]:
        """Get shapes for a specific cell"""
        if 0 <= cell_idx < len(self.camera_cells):
            return self.camera_cells[cell_idx].shapes
        return []

    def unHighlight(self):
        if self.hShape:
            self.hShape.highlightClear()
        self.hShape = None
        self.hVertex = None
        self.hEdge = None
        # Always restore cursor to default when not hovering over any shape
        self._cursor = CURSOR_DEFAULT
        self.setCursor(CURSOR_DEFAULT)
        self.update()

    def overrideCursor(self, cursor):
        """Override cursor - ensure we restore previous cursor first"""
        if self._cursor != CURSOR_DEFAULT:
            # Restore previous cursor before setting new one
            self.setCursor(CURSOR_DEFAULT)
        self._cursor = cursor
        self.setCursor(cursor)

    def restoreCursor(self):
        """Restore cursor to default - always ensure cursor is reset"""
        if self._cursor != CURSOR_DEFAULT:
            self.setCursor(CURSOR_DEFAULT)
            self._cursor = CURSOR_DEFAULT

    def drawing(self) -> bool:
        """Check if currently drawing a shape"""
        return self.mode == CanvasMode.CREATE
    
    def fillDrawing(self) -> bool:
        """Check if fill should be enabled for drawing"""
        return self._fill_drawing
    
    def _update_status(self):
        messages = []
        if self.drawing():
            messages.append(f"Creating {self._createMode}")
        else:
            messages.append("Editing shapes")
        
        if self.current_cell_index is not None:
            cell = self.camera_cells[self.current_cell_index]
            messages.append(f"Camera: {cell.camera_id}")
        
        self.statusUpdated.emit(" • ".join(messages))

    def storeShapes(self):
        """Store backup of all shapes from all cells"""
        shapesBackup = {}
        for idx, cell in enumerate(self.camera_cells):
            cell_backup = []
            for shape in cell.shapes:
                cell_backup.append(shape.copy())
            shapesBackup[idx] = cell_backup
        
        if len(self.shapesBackups) > self.num_backups:
            self.shapesBackups = self.shapesBackups[-self.num_backups - 1 :]
        self.shapesBackups.append(shapesBackup)

    @property
    def isShapeRestorable(self):
        """Check if shapes can be restored (undo is possible)"""
        if len(self.shapesBackups) < 2:
            return False
        return True

    def restoreShape(self):
        """Restore shapes from backup (undo)"""
        if not self.isShapeRestorable:
            return
        self.shapesBackups.pop()  # latest
        
        # Restore previous state
        shapesBackup = self.shapesBackups.pop()
        for idx, cell_backup in shapesBackup.items():
            if idx < len(self.camera_cells):
                self.camera_cells[idx] = self.camera_cells[idx]._replace(
                    shapes=cell_backup
                )
        
        self.selectedShapes = []
        for cell in self.camera_cells:
            for shape in cell.shapes:
                shape.selected = False
        self.update()

    def selectShapes(self, shapes: list[Shape]):
        """Select shapes"""
        self.selectedShapes = shapes
        for shape in shapes:
            shape.selected = True
        self.selectionChanged.emit(shapes)
        self.update()

    def deSelectShape(self):
        """Deselect all shapes"""
        for shape in self.selectedShapes:
            shape.selected = False
        self.selectedShapes = []
        self.selectionChanged.emit([])
        self.update()

    def setLastLabel(self, text: str, flags: dict):
        """Set label for the current shape being drawn"""
        if self.current:
            self.current.label = text
            self.current.flags = flags
        return self.current

    def undoLastLine(self):
        """Undo last point added to current shape"""
        if self.current and self.current.points:
            self.current.popPoint()

    def undoLastPoint(self):
        """Undo last point added to current shape"""
        self.undoLastLine()

    def setShapeVisible(self, shape, visible):
        self.visible[shape] = visible
        self.update()

    @property
    def shapes(self) -> list[Shape]:
        """Get all shapes from all cells (for compatibility)"""
        all_shapes = []
        for cell in self.camera_cells:
            all_shapes.extend(cell.shapes)
        return all_shapes

    @property
    def pixmap(self) -> QtGui.QPixmap:
        """Get a combined pixmap (for compatibility, returns first cell's pixmap)"""
        if self.camera_cells:
            return self.camera_cells[0].pixmap
        return QtGui.QPixmap()

    def visibleRegion(self):
        """Return visible region (for compatibility)"""
        return QtGui.QRegion(self.rect())

    def enableDragging(self, enabled: bool):
        """Enable/disable dragging (for compatibility)"""
        # Multi-camera canvas doesn't use dragging in the same way
        pass

    def loadPixmap(self, pixmap: QtGui.QPixmap, clear_shapes: bool = False):
        """Load pixmap (for compatibility, not used in multi-camera mode)"""
        # This is not used in multi-camera mode
        pass

    def adjustSize(self):
        """Adjust size (for compatibility)"""
        self.update()

    def deleteSelected(self):
        """Delete selected shapes"""
        deleted_shapes = []
        if not self.selectedShapes:
            return deleted_shapes
        
        # Find and remove shapes from their cells
        for shape in self.selectedShapes:
            for idx, cell in enumerate(self.camera_cells):
                if shape in cell.shapes:
                    cell.shapes.remove(shape)
                    deleted_shapes.append(shape)
                    self.camera_cells[idx] = cell
                    break
        
        self.storeShapes()
        self.selectedShapes = []
        self.update()
        return deleted_shapes

    def removeSelectedPoint(self):
        """Remove selected point from shape"""
        if self.hShape and self.hVertex is not None:
            if self.hShape.canRemovePoint():
                self.hShape.removePoint(self.hVertex)
                self.hVertex = None
                if not self.hShape.points:
                    # Remove shape if no points left
                    for idx, cell in enumerate(self.camera_cells):
                        if self.hShape in cell.shapes:
                            cell.shapes.remove(self.hShape)
                            self.camera_cells[idx] = cell
                            break
                    self.hShape = None
                self.update()

    def endMove(self, copy: bool = False):
        """End move operation (for compatibility with regular canvas)"""
        if copy and self.selectedShapesCopy:
            # Copy shapes - add copies to their cells
            new_shapes = []
            for shape_copy in self.selectedShapesCopy:
                # Find which cell the original shape belongs to
                for idx, cell in enumerate(self.camera_cells):
                    # Check if any selected shape is in this cell
                    if any(s in cell.shapes for s in self.selectedShapes):
                        # Add copy to same cell
                        new_shape = shape_copy.copy()
                        cell.shapes.append(new_shape)
                        self.camera_cells[idx] = cell
                        new_shapes.append(new_shape)
                        break
            
            # Update selected shapes to the new copies
            if new_shapes:
                for s in self.selectedShapes:
                    s.selected = False
                self.selectedShapes = new_shapes
                for s in self.selectedShapes:
                    s.selected = True
                self.selectionChanged.emit(self.selectedShapes)
            
            self.selectedShapesCopy = []
        else:
            # Just end move
            self.selectedShapesCopy = []
        
        self.storeShapes()
        self.repaint()
        return True

    def resetState(self):
        self.camera_cells = []
        self.current = None
        self.current_cell_index = None
        self.selectedShapes = []
        self.hShape = None
        self.shapesBackups = []
        self.movingShape = False
        self.movingVertex = False
        self._move_start_pos = None
        self._move_start_cell_idx = None
        self.update()

