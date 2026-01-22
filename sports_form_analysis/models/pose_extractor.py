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

    # Try to access the new PoseLandmarker API
    # MediaPipe changed its API in newer versions
    try:
        _test_pose = mp.tasks.vision.PoseLandmarker
        _test_options = mp.tasks.vision.PoseLandmarkerOptions
        MEDIAPIPE_AVAILABLE = True
    except AttributeError as attr_e:
        # If direct access fails, MediaPipe might not be properly installed
        # or there's a version compatibility issue
        _import_error = (
            f"MediaPipe PoseLandmarker not accessible. "
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

        # If we got here, MediaPipe is available with the new API
        try:
            BaseOptions = mp.tasks.BaseOptions
            PoseLandmarker = mp.tasks.vision.PoseLandmarker
            PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            # Create the model asset path by downloading the model file
            import tempfile
            import urllib.request
            import os

            # Download the pose landmarker model to a temporary file
            model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
            model_path = os.path.join(os.path.dirname(__file__), "assets", "pose_landmarker_full.task")

            # Create assets directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # Download the model if it doesn't exist locally
            if not os.path.exists(model_path):
                print(f"Downloading pose landmarker model to {model_path}...")
                urllib.request.urlretrieve(model_url, model_path)
                print("Model downloaded successfully.")

            self.pose_landmarker = PoseLandmarker.create_from_options(
                PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=VisionRunningMode.IMAGE,  # Use IMAGE mode for single frames
                    min_pose_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence
                )
            )

            # Store references for later use
            self.mp_vision = mp.tasks.vision
        except (AttributeError, TypeError) as e:
            mp_version = getattr(mp, '__version__', 'unknown')
            raise ImportError(
                f"Failed to initialize MediaPipe PoseLandmarker. Error: {e}. "
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

        # Convert to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Process frame
        detection_result = self.pose_landmarker.detect(mp_image)

        if not detection_result.pose_landmarks:
            return None

        # Extract keypoints
        keypoints = np.zeros((33, 3))
        for idx, landmark in enumerate(detection_result.pose_landmarks[0]):  # Take first person detected
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
        Draw pose skeleton on frame using OpenCV (since drawing_utils is not available in new API)

        Args:
            frame: Input frame
            keypoints: Keypoints array (33, 3)

        Returns:
            Annotated frame
        """
        if keypoints is None:
            return frame

        annotated_frame = frame.copy()
        h, w, _ = frame.shape

        # Define connections between keypoints (simplified version of POSE_CONNECTIONS)
        POSE_CONNECTIONS = [
            (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),  # upper body
            (23, 24), (11, 23), (12, 24),  # torso
            (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32)  # legs
        ]

        # Draw keypoints
        for idx, (x_norm, y_norm, visibility) in enumerate(keypoints):
            if visibility > 0.5:  # Only draw visible keypoints
                x = int(x_norm * w)
                y = int(y_norm * h)
                cv2.circle(annotated_frame, (x, y), 4, (0, 255, 0), -1)

        # Draw connections
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            start_point = keypoints[start_idx]
            end_point = keypoints[end_idx]

            if start_point[2] > 0.5 and end_point[2] > 0.5:  # Both keypoints visible
                start_x = int(start_point[0] * w)
                start_y = int(start_point[1] * h)
                end_x = int(end_point[0] * w)
                end_y = int(end_point[1] * h)

                cv2.line(annotated_frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)

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
        if hasattr(self, 'pose_landmarker'):
            self.pose_landmarker.close()
