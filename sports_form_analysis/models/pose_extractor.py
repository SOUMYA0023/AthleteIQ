"""
Pose Extraction Module using MediaPipe Pose
Extracts 33 keypoints per frame from video input
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

# Import MediaPipe with error handling
MEDIAPIPE_AVAILABLE = False
mp = None
_import_error = None

try:
    import mediapipe as mp
    mp_version = getattr(mp, '__version__', 'unknown')
    
    # Try to access solutions.pose directly
    # This is the standard way for MediaPipe 0.10.30+
    try:
        _test_pose = mp.solutions.pose
        _test_drawing = mp.solutions.drawing_utils
        MEDIAPIPE_AVAILABLE = True
    except AttributeError as attr_e:
        # If direct access fails, MediaPipe might not be properly installed
        # or there's a version compatibility issue
        _import_error = (
            f"MediaPipe solutions.pose not accessible. "
            f"Version: {mp_version}. "
            f"Error: {attr_e}. "
            f"Please ensure mediapipe>=0.10.30 is correctly installed."
        )
        MEDIAPIPE_AVAILABLE = False
except ImportError as e:
    _import_error = f"MediaPipe import failed: {e}. Please install with: pip install mediapipe>=0.10.30"
    MEDIAPIPE_AVAILABLE = False
except Exception as e:
    _import_error = f"MediaPipe initialization error: {e}"
    MEDIAPIPE_AVAILABLE = False


class PoseExtractor:
    """
    Extracts human pose keypoints from video frames using MediaPipe Pose
    """
    
    def __init__(self, min_detection_confidence: float = 0.5, 
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe Pose model
        
        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
        """
        if not MEDIAPIPE_AVAILABLE or mp is None:
            error_msg = _import_error if _import_error else "MediaPipe is not installed"
            raise ImportError(
                f"{error_msg}. Please install it with: pip install mediapipe>=0.10.30"
            )
        
        # If we got here, MediaPipe is available and solutions.pose is accessible
        try:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                model_complexity=2  # Use full model for better accuracy
            )
            self.mp_drawing = mp.solutions.drawing_utils
        except (AttributeError, TypeError) as e:
            mp_version = getattr(mp, '__version__', 'unknown')
            raise ImportError(
                f"Failed to initialize MediaPipe Pose. Error: {e}. "
                f"MediaPipe version: {mp_version}. "
                f"This might be a version compatibility issue. "
                f"Try: pip install --upgrade mediapipe>=0.10.30"
            )
        
    def extract_keypoints(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract pose keypoints from a single frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Array of shape (33, 3) containing (x, y, visibility) for each keypoint
            Returns None if no pose detected
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Process frame
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
        
        # Extract keypoints
        keypoints = np.zeros((33, 3))
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            keypoints[idx] = [landmark.x, landmark.y, landmark.visibility]
        
        return keypoints
    
    def extract_from_video(self, video_path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Extract keypoints from entire video
        
        Args:
            video_path: Path to input video file
            
        Returns:
            Tuple of (list of keypoint arrays, list of frames)
        """
        cap = cv2.VideoCapture(video_path)
        keypoints_list = []
        frames_list = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            keypoints = self.extract_keypoints(frame)
            if keypoints is not None:
                keypoints_list.append(keypoints)
                frames_list.append(frame)
            else:
                # If no pose detected, append zeros to maintain frame alignment
                keypoints_list.append(np.zeros((33, 3)))
                frames_list.append(frame)
        
        cap.release()
        return keypoints_list, frames_list
    
    def draw_pose(self, frame: np.ndarray, keypoints: Optional[np.ndarray]) -> np.ndarray:
        """
        Draw pose skeleton on frame using MediaPipe
        
        Args:
            frame: Input frame
            keypoints: Keypoints array (33, 3)
            
        Returns:
            Annotated frame
        """
        if keypoints is None:
            return frame
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Create MediaPipe landmark objects
        landmarks = self.mp_pose.PoseLandmark
        
        # Create a results-like object for drawing
        class PoseLandmark:
            def __init__(self, x, y, z, visibility):
                self.x = x
                self.y = y
                self.z = z
                self.visibility = visibility
        
        class PoseLandmarks:
            def __init__(self, keypoints):
                self.landmark = [PoseLandmark(kp[0], kp[1], 0, kp[2]) for kp in keypoints]
        
        class Results:
            def __init__(self, keypoints):
                self.pose_landmarks = PoseLandmarks(keypoints)
        
        results = Results(keypoints)
        
        # Draw pose
        annotated_frame = frame.copy()
        rgb_annotated = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        rgb_annotated.flags.writeable = True
        
        self.mp_drawing.draw_landmarks(
            rgb_annotated,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )
        
        # Convert back to BGR
        annotated_frame = cv2.cvtColor(rgb_annotated, cv2.COLOR_RGB2BGR)
        
        return annotated_frame
    
    def normalize_keypoints(self, keypoints: np.ndarray, 
                           frame_width: int, frame_height: int) -> np.ndarray:
        """
        Normalize keypoints to account for camera distance
        Uses body scale (shoulder-to-hip distance) for normalization
        
        Args:
            keypoints: Keypoints array (33, 3)
            frame_width: Frame width
            frame_height: Frame height
            
        Returns:
            Normalized keypoints array
        """
        # Convert pixel coordinates to normalized [0, 1]
        normalized = keypoints.copy()
        normalized[:, 0] = keypoints[:, 0]  # Already normalized by MediaPipe
        normalized[:, 1] = keypoints[:, 1]  # Already normalized by MediaPipe
        
        # Calculate body scale using shoulder-to-hip distance
        left_shoulder = keypoints[11]  # Left shoulder
        right_shoulder = keypoints[12]  # Right shoulder
        left_hip = keypoints[23]  # Left hip
        right_hip = keypoints[24]  # Right hip
        
        # Average shoulder and hip positions
        shoulder_center = (left_shoulder[:2] + right_shoulder[:2]) / 2
        hip_center = (left_hip[:2] + right_hip[:2]) / 2
        
        # Body scale (distance between shoulder and hip centers)
        body_scale = np.linalg.norm(shoulder_center - hip_center)
        
        if body_scale > 0:
            # Normalize by body scale
            normalized[:, :2] = normalized[:, :2] / body_scale
        
        return normalized
    
    def close(self):
        """Release MediaPipe resources"""
        self.pose.close()
