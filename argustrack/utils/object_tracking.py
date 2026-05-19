from collections import OrderedDict
from collections import deque
from typing import Dict, List, Tuple, Optional
import numpy as np
import scipy
import scipy.linalg
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack(object):
    _count = 0

    track_id: int = 0
    is_activated: bool = False
    state: int = TrackState.New

    history: OrderedDict = OrderedDict()
    features: List = []  # type: ignore
    curr_feature: Optional[np.ndarray] = None
    score: float = 0.0
    start_frame: int = 0
    frame_id: int = 0
    time_since_update: int = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self) -> int:
        return self.frame_id

    @staticmethod
    def next_id() -> int:
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.
    Adapted for 2D BEV tracking (x, y, vx, vy).

    The 4-dimensional state space
        x, y, vx, vy
    contains the bounding box center position (x, y) and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y) is taken as direct observation of the state space (linear observation model).
    """

    def __init__(self):
        ndim, dt = 2, 1.0

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim, dtype=np.float32)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim, dtype=np.float32)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model.
        # Reduced position uncertainty for better tracking of stationary objects
        self._std_weight_position = 1.0 / 20
        # Increased velocity uncertainty to allow better adaptation when objects stop
        self._std_weight_velocity = 1.0 / 80

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y) with center position (x, y).

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (4 dimensional) and covariance matrix (4x4
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel].astype(np.float32)

        std = [
            2 * self._std_weight_position,
            2 * self._std_weight_position,
            10 * self._std_weight_velocity,
            10 * self._std_weight_velocity
        ]
        covariance = np.diag(np.square(std)).astype(np.float32)
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 4 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 4x4 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state.
        """
        std_pos = [self._std_weight_position, self._std_weight_position]
        std_vel = [self._std_weight_velocity, self._std_weight_velocity]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])).astype(np.float32)

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (4 dimensional array).
        covariance : ndarray
            The state's covariance matrix (4x4 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.
        """
        std = [self._std_weight_position, self._std_weight_position]
        innovation_cov = np.diag(np.square(std)).astype(np.float32)

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run Kalman filter prediction step (Vectorized version).
        Parameters
        ----------
        mean : ndarray
            The Nx4 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx4x4 dimensional covariance matrices of the object states at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state.
        """
        ones = np.ones_like(mean[:, 0])
        std_pos = [self._std_weight_position * ones, self._std_weight_position * ones]
        std_vel = [self._std_weight_velocity * ones, self._std_weight_velocity * ones]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov).astype(np.float32)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (4 dimensional).
        covariance : ndarray
            The state's covariance matrix (4x4 dimensional).
        measurement : ndarray
            The 2 dimensional measurement vector (x, y), where (x, y)
            is the center position.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(
        self, 
        mean: np.ndarray, 
        covariance: np.ndarray, 
        measurements: np.ndarray, 
        only_position: bool = False, 
        metric: str = 'maha'
    ) -> np.ndarray:
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (4 dimensional).
        covariance : ndarray
            Covariance of the state distribution (4x4 dimensional).
        measurements : ndarray
            An Nx2 dimensional matrix of N measurements, each in
            format (x, y) where (x, y) is the bounding box center position.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        metric : str
            Distance metric to use ('maha' for Mahalanobis, 'gaussian' for Euclidean)
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')


def linear_assignment(cost_matrix: np.ndarray, thresh: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear assignment using Hungarian algorithm with threshold.
    
    Parameters
    ----------
    cost_matrix : ndarray
        Cost matrix of shape (N, M)
    thresh : float
        Maximum cost threshold for valid matches
        
    Returns
    -------
    matches : ndarray
        Array of shape (K, 2) with matched pairs [row, col]
    unmatched_a : ndarray
        Unmatched indices from first set
    unmatched_b : ndarray
        Unmatched indices from second set
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int), 
            np.array(range(cost_matrix.shape[0])), 
            np.array(range(cost_matrix.shape[1]))
        )
    
    # Use scipy's linear_sum_assignment (Hungarian algorithm)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Filter by threshold
    valid_mask = cost_matrix[row_ind, col_ind] <= thresh
    matches = np.column_stack([row_ind[valid_mask], col_ind[valid_mask]])
    
    # Find unmatched
    matched_rows = set(matches[:, 0]) if len(matches) > 0 else set()
    matched_cols = set(matches[:, 1]) if len(matches) > 0 else set()
    
    unmatched_a = np.array([i for i in range(cost_matrix.shape[0]) if i not in matched_rows])
    unmatched_b = np.array([i for i in range(cost_matrix.shape[1]) if i not in matched_cols])
    
    return matches, unmatched_a, unmatched_b


def center_distance(atracks: List[np.ndarray], btracks: List[np.ndarray]) -> np.ndarray:
    """
    Compute cost based on center point distance
    :type atracks: list[np.ndarray] | list[Tuple[float, float]]
    :type btracks: list[np.ndarray] | list[Tuple[float, float]]

    :rtype cost_matrix np.ndarray
    """
    cost_matrix = np.zeros((len(atracks), len(btracks)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix

    atracks = np.stack(atracks)
    btracks = np.stack(btracks)

    cost_matrix = cdist(atracks, btracks, 'euclidean')

    return cost_matrix


def joint_stracks(tlista: List, tlistb: List) -> List:
    """Join two track lists, avoiding duplicates by track_id."""
    exists: Dict[int, int] = {}
    res: List = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista: List, tlistb: List) -> List:
    """Subtract tlistb from tlista based on track_id."""
    stracks: Dict[int, object] = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa: List, stracksb: List) -> Tuple[List, List]:
    """Remove duplicate tracks between two lists."""
    track_a = [t.xy_prev for t in stracksa]
    track_b = [t.xy for t in stracksb]
    pdist = center_distance(track_a, track_b)
    pairs = np.where(pdist < 6)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, xy: np.ndarray, xy_prev: np.ndarray, score: float, buffer_size: int = 30):
        # wait activate
        self._xy = xy
        self._xy_prev = xy_prev
        self.kalman_filter: Optional[KalmanFilter] = None
        self.mean: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat: Optional[np.ndarray] = None
        self.features: deque = deque([], maxlen=buffer_size)  # type: ignore
        self.alpha: float = 0.9

    def update_features(self, feat: np.ndarray):
        """Update features with exponential moving average."""
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        """Predict next state using Kalman filter."""
        if self.mean is not None and self.covariance is not None:
            mean_state = self.mean.copy()
            self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks: List['STrack']):
        """Vectorized prediction for multiple tracks."""
        if len(stracks) > 0:
            # Filter out tracks with None mean/covariance
            valid_stracks: List[STrack] = []
            valid_means: List[np.ndarray] = []
            valid_covs: List[np.ndarray] = []
            for st in stracks:
                if st.mean is not None and st.covariance is not None:
                    valid_stracks.append(st)
                    valid_means.append(st.mean.copy())
                    valid_covs.append(st.covariance)
            if len(valid_stracks) > 0:
                multi_mean = np.asarray(valid_means)
                multi_covariance = np.asarray(valid_covs)
                multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
                for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                    valid_stracks[i].mean = mean
                    valid_stracks[i].covariance = cov

    def activate(self, kalman_filter: KalmanFilter, frame_id: int):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.xy)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: 'STrack', frame_id: int, new_id: bool = False):
        """Re-activate a lost track."""
        if self.kalman_filter is not None and self.mean is not None and self.covariance is not None:
            # Use current position (_xy) instead of predicted position (xy)
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, new_track._xy
            )

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track: 'STrack', frame_id: int, update_feature: bool = False):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        if self.kalman_filter is not None and self.mean is not None and self.covariance is not None:
            # Use current position (_xy) instead of predicted position (xy)
            measurement = new_track._xy
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, measurement
            )
            # If object appears stationary (very small movement), reduce velocity estimate
            if self.tracklet_len > 1:
                velocity = self.mean[2:4]
                velocity_magnitude = np.linalg.norm(velocity)
                if velocity_magnitude < 0.1:  # Object appears stationary (< 0.1 m/s)
                    # Decay velocity estimate towards zero to prevent drift
                    self.mean[2:4] = self.mean[2:4] * 0.5
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    def xy(self) -> np.ndarray:
        """Get current position."""
        if self.mean is None:
            return self._xy
        return self.mean[:2]

    @property
    def xy_prev(self) -> np.ndarray:
        """Get previous position."""
        return self._xy_prev

    def __repr__(self) -> str:
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class JDETracker:
    """JDE-style tracker for object tracking."""
    
    def __init__(self, conf_thres: float = 0.1, track_buffer: int = 5):
        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []

        self.frame_id = 0
        self.det_thresh = conf_thres
        self.max_time_lost = track_buffer

        self.kalman_filter = KalmanFilter()

    def update(
        self, 
        dets: np.ndarray, 
        dets_prev: np.ndarray, 
        score: np.ndarray
    ) -> List[STrack]:
        """Update tracker with new detections.
        
        Parameters
        ----------
        dets : ndarray
            Current detections of shape (N, 2) with (x, y) coordinates
        dets_prev : ndarray
            Previous detections of shape (N, 2) with (x, y) coordinates
        score : ndarray
            Detection scores of shape (N,)
            
        Returns
        -------
        List[STrack]
            List of active tracked objects
        """
        self.frame_id += 1
        activated_starcks: List[STrack] = []
        refind_stracks: List[STrack] = []
        lost_stracks: List[STrack] = []
        removed_stracks: List[STrack] = []

        remain_inds = score > self.det_thresh - 0.1
        dets = dets[remain_inds]
        dets_prev = dets_prev[remain_inds]
        score = score[remain_inds]

        if len(dets) > 0:
            """Detections"""
            detections = [
                STrack(xy, xy_prev, s, self.max_time_lost) 
                for (xy, xy_prev, s) in zip(dets, dets_prev, score)
            ]
        else:
            detections = []

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed: List[STrack] = []
        tracked_stracks: List[STrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: Association'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        strack_pool_xy = [track.xy for track in strack_pool]
        # Use current position instead of previous position for better matching
        detections_xy = [det._xy for det in detections]

        dists = center_distance(strack_pool_xy, detections_xy)
        # Reduced threshold for better matching when objects are stationary
        # Use adaptive threshold: smaller for tracked, larger for lost tracks
        matches, u_track, u_detection = linear_assignment(dists, thresh=30)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = strack_pool[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        # Use current position instead of previous position
        detections_xy = [det._xy for det in detections]
        unconfirmed_xy = [track.xy for track in unconfirmed]
        dists = center_distance(unconfirmed_xy, detections_xy)
        # Reduced threshold for unconfirmed tracks
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=50)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 3: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
            
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


class ObjectTracking:
    """
    Object tracking using JDETracker approach.
    Runs a JDE-style tracker on per-frame (x, y) detections.
    
    Note: Uses actual detection points (footpoints) from clusters, not computed centroids.
    Each point in list_of_labels should be a representative footpoint from a cluster.
    """

    def __init__(
        self,
        list_of_labels: List[List[Tuple[float, float]]],
        max_distance: float = 9.0,
        max_age: int = 10,
        min_hits: int = 2,
        process_noise: float = 1.0,
        meas_noise: float = 0.5,
        dt: float = 1.0,
        conf_thres: float = 0.1,
        track_buffer: int = 5,
    ):
        """
        Initialize ObjectTracking.
        
        Parameters
        ----------
        list_of_labels : List[List[Tuple[float, float]]]
            List of frames, each containing list of (x, y) detection coordinates
        max_distance : float
            Maximum distance for association (not used directly in JDETracker, kept for compatibility)
        max_age : int
            Maximum age for tracks (mapped to track_buffer)
        min_hits : int
            Minimum hits to confirm track (not used directly, kept for compatibility)
        process_noise : float
            Process noise (not used directly, kept for compatibility)
        meas_noise : float
            Measurement noise (not used directly, kept for compatibility)
        dt : float
            Time step (not used directly, kept for compatibility)
        conf_thres : float
            Confidence threshold for detections
        track_buffer : int
            Maximum frames a track can be lost before removal
        """
        self.list_of_labels = list_of_labels
        self.global_id: Dict[Tuple[int, int], int] = {}
        # Use track_buffer as max_time_lost
        self.tracker = JDETracker(
            conf_thres=conf_thres,
            track_buffer=max_age if max_age > 0 else track_buffer,
        )
        self.prev_detections: Optional[np.ndarray] = None

    def assign_global_id(self) -> Dict[Tuple[int, int], int]:
        """
        Run the JDETracker and return mapping (frame_idx, obj_idx) -> global_id.
        
        Returns
        -------
        Dict[Tuple[int, int], int]
            Mapping from (frame_idx, obj_idx) to global track ID
        """
        self.global_id = {}
        self.prev_detections = None
        
        for frame_idx, detections in enumerate(self.list_of_labels):
            try:
                if len(detections) == 0:
                    dets = np.empty((0, 2), dtype=np.float32)
                    dets_prev = np.empty((0, 2), dtype=np.float32)
                    score = np.empty((0,), dtype=np.float32)
                else:
                    dets = np.array(detections, dtype=np.float32)
                    # Use previous frame detections if available, otherwise use current
                    if self.prev_detections is not None and len(self.prev_detections) == len(dets):
                        dets_prev = self.prev_detections
                    else:
                        # For first frame or when count changes, use current position as previous
                        dets_prev = dets.copy()
                    # Use default score of 1.0 for all detections
                    score = np.ones(len(dets), dtype=np.float32)
                
                # Update tracker
                output_stracks = self.tracker.update(dets, dets_prev, score)
                
                # Map detections to track IDs
                # Use Hungarian algorithm for optimal matching, prioritizing tracks with longer history
                if len(dets) > 0 and len(output_stracks) > 0:
                    # Get track positions
                    track_positions = np.array([track.xy for track in output_stracks])
                    
                    # Build cost matrix: distance + penalty for short tracks (to prioritize long tracks)
                    distances = cdist(dets, track_positions, metric='euclidean')
                    
                    # Add penalty for short tracks to prioritize tracks with longer history
                    # This helps prevent ID switches when objects are stationary
                    cost_matrix = distances.copy()
                    for track_idx, track in enumerate(output_stracks):
                        track_age = track.tracklet_len
                        # Reduce cost for older tracks (prefer keeping same ID)
                        if track_age > 0:
                            cost_matrix[:, track_idx] *= (1.0 - min(0.3, track_age / 30.0))
                    
                    # Use Hungarian algorithm for optimal assignment
                    if distances.size > 0:
                        matches, u_det, u_track = linear_assignment(cost_matrix, thresh=15.0)
                        
                        # Assign matched detections
                        for det_idx, track_idx in matches:
                            track = output_stracks[track_idx]
                            self.global_id[(frame_idx, det_idx)] = track.track_id
                
                # Update previous detections for next frame
                if len(dets) > 0:
                    self.prev_detections = dets.copy()
                else:
                    self.prev_detections = None
                    
            except Exception as e:
                import traceback
                print(f"Error in frame {frame_idx}: {e}")
                print(traceback.format_exc())
                continue
                
        return self.global_id

    def get_global_id(self, frame_idx: int, obj_idx: int) -> int:
        """
        Get global ID for a specific frame and object index.
        
        Parameters
        ----------
        frame_idx : int
            Frame index
        obj_idx : int
            Object index within frame
            
        Returns
        -------
        int
            Global track ID, or -1 if not found
        """
        return self.global_id.get((frame_idx, obj_idx), -1)
