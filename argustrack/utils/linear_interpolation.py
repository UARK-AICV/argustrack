import numpy as np
import cv2

def vector_lerp(p0, p1, t):
    return (1 - t) * p0 + t * p1

def get_camera_angles(rvec):
    if rvec is None:
        return {'elevation': 0.0, 'azimuth': 0.0, 'is_top_down': False, 'is_side_view': False}
    
    rvec = np.array(rvec).flatten()[:3].reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    camera_z = R[:, 2]
    elevation_rad = np.pi/2 - np.arccos(np.clip(np.abs(camera_z[2]), 0, 1))
    elevation_deg = np.degrees(elevation_rad)
    camera_z_xy = camera_z[:2]
    if np.linalg.norm(camera_z_xy) > 1e-6:
        azimuth_rad = np.arctan2(camera_z_xy[1], camera_z_xy[0])
        azimuth_deg = np.degrees(azimuth_rad) % 360
    else:
        azimuth_deg = 0.0
    
    is_top_down = elevation_deg >= 60.0
    is_side_view = min(abs(azimuth_deg - a) for a in [0, 90, 180, 270, 360]) <= 30.0
    return {
        'elevation': elevation_deg,
        'azimuth': azimuth_deg,
        'is_top_down': is_top_down,
        'is_side_view': is_side_view
    }

def smooth_step(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def get_dynamic_projection_point(bbox, distance_to_camera, rvec=None, min_dist=1.0, max_dist=10.0, 
                                 img_width=None, img_height=None):
    x_min, y_min, x_max, y_max = bbox
    w, h = x_max - x_min, y_max - y_min
    
    # Normalize aspect ratio based on full frame size to keep coordinates consistent.
    # If img_width and img_height are given, normalize with respect to frame size.
    # Otherwise, fall back to using absolute bbox size (backward compatibility).
    if img_width is not None and img_height is not None and img_width > 0 and img_height > 0:
        # Normalize bbox size with respect to frame size
        w_normalized = w / float(img_width)
        h_normalized = h / float(img_height)
        # Aspect ratio normalized by frame aspect ratio
        # aspect_ratio = (h_normalized / w_normalized) * (img_width / img_height)
        # Or simply: aspect_ratio = h / w (absolute) but normalized by frame
        frame_aspect = img_height / float(img_width)
        bbox_aspect = h / max(w, 1)
        # Normalize aspect ratio: bbox_aspect / frame_aspect
        aspect_ratio = bbox_aspect / frame_aspect
    else:
        # Fallback: use absolute aspect ratio
        aspect_ratio = h / max(w, 1)
    
    # Round aspect_ratio to reduce sensitivity to small bbox changes.
    # Round to 0.1 to be more stable across frames.
    aspect_ratio = round(aspect_ratio * 10) / 10.0

    points = {
        'bottom': np.array([(x_min + x_max) / 2, y_max]),
        'center': np.array([(x_min + x_max) / 2, y_min + h * 0.5])
    }

    cam = get_camera_angles(rvec) if rvec is not None else {'elevation': 0.0, 'is_side_view': False}
    elev = np.clip(cam['elevation'], 0.0, 90.0)

    denom = max(max_dist - min_dist, 1e-6)
    normalized = np.clip((distance_to_camera - min_dist) / denom, 0.0, 1.0)
    t_dist = smooth_step(normalized)

    cos_weight = np.cos(np.radians(np.clip(elev, 0.0, 90.0)))
    
    # Handle NaN or invalid values
    if not np.isfinite(cos_weight) or cos_weight < 0:
        cos_weight = 0.0

    t_angle = np.clip(cos_weight ** 0.7, 0.0, 1.0)
    
    # Ensure t_angle is finite
    if not np.isfinite(t_angle):
        t_angle = 0.0

    t_total = np.clip((t_angle * (0.9 * t_dist + 0.1)), 0.0, 1.0)

    # proj_by_angle = vector_lerp(points['center'], points['bottom'], t_angle)
    proj_point = vector_lerp(points['center'], points['bottom'], t_total)

    if aspect_ratio > 1.6: 
        proj_point[1] += -h * 0.05 * min((aspect_ratio - 1.6), 1.0)
    elif aspect_ratio < 0.6: 
        proj_point[1] += h * 0.04

    proj_point[0] = np.clip(proj_point[0], x_min, x_max)
    proj_point[1] = np.clip(proj_point[1], y_min, y_max)

    return proj_point, {
        'elevation': elev,
        't_angle': t_angle,
        't_dist': t_dist,
        'aspect_ratio': aspect_ratio
    }

def estimate_distance_from_bbox_height(bbox, camera_matrix, average_object_height=2):
    """Estimate distance based on bbox height and known object height"""
    _, y_min, _, y_max = bbox
    h_pixels = max(y_max - y_min, 1)
    return (average_object_height * camera_matrix[1, 1]) / h_pixels

def estimate_distance_from_camera(bbox_center_px, camera_extrinsics, camera_intrinsics, ground_z=0):
    R = camera_extrinsics["R"]
    t = camera_extrinsics["t"]
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    x_norm = (bbox_center_px[0] - cx) / fx
    y_norm = (bbox_center_px[1] - cy) / fy

    ray_cam = np.array([x_norm, y_norm, 1.0])
    ray_world = R.T @ ray_cam
    cam_pos = -R.T @ t
    if ray_world[2] == 0:
        return np.inf
    s = (ground_z - cam_pos[2]) / ray_world[2]
    ground_point = cam_pos + s * ray_world
    distance = np.linalg.norm(ground_point - cam_pos)
    # print(distance)
    return distance