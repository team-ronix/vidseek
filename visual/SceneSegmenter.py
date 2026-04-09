from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from datastructures import Scene
import cv2


class SceneSegmenter:
    def __init__(self, video_path, default_threshold=20):
        """
        Initialize SceneSegmenter.
        
        Args:
            video_path: Path to video file
            default_threshold: Default threshold for scene detection (0-255, higher = less sensitive)
        """
        self.video_path = video_path
        self.default_threshold = default_threshold
        self.scenes_list = []
        self.middle_frames = []

    def segment_video(self, threshold=None):
        """
        Segment video into scenes based on content changes.
        
        Args:
            threshold: Content detection threshold (0-255). If None, uses default_threshold.
                      Higher values = less sensitive (fewer scene cuts)
                      Lower values = more sensitive (more scene cuts)
        """
        if threshold is None:
            threshold = self.default_threshold
        
        # Clear previous scenes
        self.scenes_list = []
        self.middle_frames = []
        
        video = open_video(self.video_path)
        scene_manager = SceneManager()
        # If difference between two consecutive frames > threshold -> scene boundary
        scene_manager.add_detector(ContentDetector(threshold=threshold))

        scene_manager.detect_scenes(video)
        scenes = scene_manager.get_scene_list()
        
        for i, (start, end) in enumerate(scenes):
            scene = Scene(
                index=i,
                start_time=start.get_seconds(),
                end_time=end.get_seconds(),
                start_frame=start.get_frames(),
                end_frame=end.get_frames(),
                fps=start.get_framerate()
            )
            self.scenes_list.append(scene)
        
        return self.scenes_list
    
    def calculate_frames_to_extract(self, scene):
        """
        Calculate which frames to extract based on scene duration.
        
        Logic:
        - Short scenes (< 2s): 1 frame (middle)
        - Medium scenes (2-5s): 2 frames (at 33% and 67%)
        - Long scenes (5-10s): 3 frames (at 25%, 50%, 75%)
        - Very long scenes (> 10s): 1 frame every 3 seconds, max 5 frames
        
        Args:
            scene: Scene object
            
        Returns:
            List of frame numbers to extract
        """
        duration = scene.duration
        start_frame = scene.start_frame
        end_frame = scene.end_frame
        total_frames = end_frame - start_frame
        
        # Handle edge case of zero or negative frames
        if total_frames <= 0:
            return [start_frame]
        
        if duration < 2.0:
            # Short scene: just middle frame
            middle = start_frame + total_frames // 2
            return [middle]
        
        elif duration < 5.0:
            # Medium scene: 2 frames at 33% and 67%
            frame1 = start_frame + int(total_frames * 0.33)
            frame2 = start_frame + int(total_frames * 0.67)
            return [frame1, frame2]
        
        elif duration < 10.0:
            # Long scene: 3 frames at 25%, 50%, 75%
            frame1 = start_frame + int(total_frames * 0.25)
            frame2 = start_frame + int(total_frames * 0.50)
            frame3 = start_frame + int(total_frames * 0.75)
            return [frame1, frame2, frame3]
        
        else:
            # Very long scene: 1 frame every 3 seconds, max 5 frames
            fps = scene.fps
            frames_per_interval = int(fps * 3)  # 3 seconds worth of frames
            
            # Safety check: prevent division by zero
            if frames_per_interval <= 0:
                frames_per_interval = 1
            
            num_frames = min(5, max(1, total_frames // frames_per_interval))
            
            # Distribute frames evenly
            frames = []
            for i in range(num_frames):
                position = (i + 1) / (num_frames + 1)
                frame_num = start_frame + int(total_frames * position)
                frames.append(frame_num)
            
            return frames
    
    def extract_frames(self):
        """
        Extract frames from each detected scene using intelligent sampling.
        Number of frames depends on scene duration.
        
        Returns:
            List of dicts with frame data
        """
        if not self.scenes_list:
            print("No scenes detected. Run segment_video() first.")
            return []
        
        self.middle_frames = []
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return []
        
        total_frames_extracted = 0
        
        for scene in self.scenes_list:
            # Calculate which frames to extract for this scene
            frame_numbers = self.calculate_frames_to_extract(scene)
            
            for frame_num in frame_numbers:
                # Set video position to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if ret:
                    self.middle_frames.append({
                        'scene_index': scene.index,
                        'frame_number': frame_num,
                        'frame': frame,
                        'scene': scene,
                        'frame_count_in_scene': len(frame_numbers)
                    })
                    total_frames_extracted += 1
                else:
                    print(f"Warning: Could not read frame {frame_num} for scene {scene.index}")
        
        cap.release()
        print(f"Extracted {total_frames_extracted} frames from {len(self.scenes_list)} scenes")
        return self.middle_frames
    
    def get_frames_for_ocr(self):
        """
        Get array of frames ready for OCR processing.
        
        Returns:
            List of frame arrays (numpy arrays in BGR format)
        """
        if not self.middle_frames:
            self.extract_frames()
        
        return [frame_data['frame'] for frame_data in self.middle_frames]
    
    def get_frames_with_metadata(self):
        """
        Get frames with their associated scene metadata.
        
        Returns:
            List of dicts with 'scene_index', 'frame_number', 'frame', 'scene', and 'frame_count_in_scene'
        """
        if not self.middle_frames:
            self.extract_frames()
        
        return self.middle_frames
    
    def get_scenes(self):
        """Get list of detected scenes."""
        return self.scenes_list
    
    def get_scene_count(self):
        """Get number of detected scenes."""
        return len(self.scenes_list)
    
    def print_scenes(self):
        """Print all detected scenes with frame extraction info."""
        print(f"\nDetected {len(self.scenes_list)} scenes:")
        print("-" * 80)
        for scene in self.scenes_list:
            duration = scene.duration
            frame_count = len(self.calculate_frames_to_extract(scene))
            frames_info = f"{frame_count} frame{'s' if frame_count > 1 else ''}"
            print(f"{scene.index:3d} | {scene.start_time:8.2f}s -> {scene.end_time:8.2f}s | "
                  f"Duration: {duration:5.2f}s | Extract: {frames_info}")