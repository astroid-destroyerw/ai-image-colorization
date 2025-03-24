import cv2
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import os
from typing import Optional, Tuple, List

class VideoColorizer:
    def __init__(self, model, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        
    def process_video(
        self,
        input_path: str,
        output_path: str,
        batch_size: int = 8,
        frame_skip: int = 1,
        progress_callback: Optional[callable] = None
    ) -> Tuple[str, List[float]]:
        """
        Process a video file and colorize its frames.
        
        Args:
            input_path: Path to input video file
            output_path: Path to save output video
            batch_size: Number of frames to process in each batch
            frame_skip: Number of frames to skip between processing
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Tuple of (output_path, list of processing times per frame)
        """
        # Open video file
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Initialize batch
        frames = []
        frame_positions = []
        processing_times = []
        
        # Process frames
        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if needed
                if len(frames) % frame_skip != 0:
                    frames.append(frame)
                    frame_positions.append(cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)
                    continue
                
                # Convert frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(gray)
                frame_positions.append(cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)
                
                # Process batch when full
                if len(frames) >= batch_size:
                    colorized_frames, batch_times = self._process_batch(frames)
                    processing_times.extend(batch_times)
                    
                    # Write frames to output video
                    for i, colorized in enumerate(colorized_frames):
                        out.write(colorized)
                        if progress_callback:
                            progress_callback(frame_positions[i], total_frames)
                        pbar.update(1)
                    
                    # Clear batch
                    frames = []
                    frame_positions = []
        
        # Process remaining frames
        if frames:
            colorized_frames, batch_times = self._process_batch(frames)
            processing_times.extend(batch_times)
            
            for i, colorized in enumerate(colorized_frames):
                out.write(colorized)
                if progress_callback:
                    progress_callback(frame_positions[i], total_frames)
                pbar.update(1)
        
        # Release resources
        cap.release()
        out.release()
        
        return output_path, processing_times
    
    def _process_batch(self, frames: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float]]:
        """
        Process a batch of frames.
        
        Args:
            frames: List of grayscale frames
            
        Returns:
            Tuple of (list of colorized frames, list of processing times)
        """
        # Convert frames to tensors
        batch = torch.stack([
            torch.from_numpy(frame).float().unsqueeze(0).unsqueeze(0)
            for frame in frames
        ]).to(self.device)
        
        # Normalize
        batch = (batch - 0.5) / 0.5
        
        # Process frames
        with torch.no_grad():
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            output = self.model(batch)
            end_time.record()
            
            torch.cuda.synchronize()
            processing_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        
        # Convert back to numpy
        output = output.cpu().numpy()
        output = (output * 0.5 + 0.5) * 255
        output = output.astype(np.uint8)
        
        # Convert to BGR format
        colorized_frames = []
        for i in range(len(output)):
            frame = output[i, 0]
            colorized = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)
            colorized_frames.append(colorized)
        
        return colorized_frames, [processing_time / len(frames)] * len(frames)

def extract_frames(video_path: str, output_dir: str, frame_skip: int = 1) -> List[str]:
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        frame_skip: Number of frames to skip between extractions
        
    Returns:
        List of paths to extracted frames
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_skip == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            
        frame_count += 1
    
    cap.release()
    return frame_paths

def create_video_from_frames(frame_paths: List[str], output_path: str, fps: int = 30) -> str:
    """
    Create a video from a list of frame paths.
    
    Args:
        frame_paths: List of paths to frame images
        output_path: Path to save output video
        fps: Frames per second
        
    Returns:
        Path to output video
    """
    if not frame_paths:
        raise ValueError("No frames provided")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_paths[0])
    height, width = first_frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        out.write(frame)
    
    out.release()
    return output_path 