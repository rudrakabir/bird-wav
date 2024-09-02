import cv2
import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine
from pydub.effects import low_pass_filter

def extract_bird_coordinates(video_path):
    cap = cv2.VideoCapture(video_path)
    backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    bird_coordinates = []
    frame_count = 0
    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgMask = backSub.apply(gray)
        _, thresh = cv2.threshold(fgMask, 244, 255, cv2.THRESH_BINARY)
        
        # Calculate frame difference
        if prev_frame is not None:
            frame_diff = cv2.absdiff(gray, prev_frame)
            _, motion_mask = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
            thresh = cv2.bitwise_and(thresh, motion_mask)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        frame_coordinates = []
        for contour in contours:
            if cv2.contourArea(contour) > 200:  # Increased minimum area
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    # Check distance from other detected birds
                    if all(np.sqrt((x-cX)**2 + (y-cY)**2) > 30 for x, y in frame_coordinates):
                        frame_coordinates.append((cX, cY))
        
        bird_coordinates.append(frame_coordinates)
        prev_frame = gray
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames, detected {len(frame_coordinates)} birds")

    cap.release()
    print(f"Total frames processed: {frame_count}")
    return bird_coordinates

def coordinates_to_frequency(x, y, width, height):
    min_freq, max_freq = 150, 450
    freq = min_freq + (x / width) * (max_freq - min_freq)
    return freq

def generate_audio(coordinates, width, height, fps, video_duration):
    total_duration_ms = int(video_duration * 1000)
    audio = AudioSegment.silent(duration=total_duration_ms)
    frame_duration = 1000 // fps
    
    for frame_num, frame_coords in enumerate(coordinates):
        frame_start = frame_num * frame_duration
        frame_end = min((frame_num + 1) * frame_duration, total_duration_ms)
        frame_audio = AudioSegment.silent(duration=frame_end - frame_start)
        
        if frame_num % 30 == 0:
            print(f"Processing frame {frame_num}/{len(coordinates)}, {len(frame_coords)} birds detected")
        
        for x, y in frame_coords:
            freq = coordinates_to_frequency(x, y, width, height)
            tone = Sine(freq).to_audio_segment(duration=frame_end - frame_start)
            tone = low_pass_filter(tone, 1000)
            frame_audio = frame_audio.overlay(tone, gain_during_overlay=-6)
        
        audio = audio.overlay(frame_audio, position=frame_start)
        
        if frame_num % fps == 0:
            print(f"  Processed {frame_num // fps} seconds, audio duration: {len(audio)}ms")
    
    bg_noise = AudioSegment.silent(duration=total_duration_ms).overlay(
        Sine(80).to_audio_segment(duration=total_duration_ms), gain_during_overlay=-20
    )
    audio = audio.overlay(bg_noise)
    
    print(f"Final audio duration: {len(audio)}ms")
    return audio

def main(video_path, output_path):
    print("Extracting bird coordinates...")
    coordinates = extract_bird_coordinates(video_path)
    print(f"Extracted coordinates from {len(coordinates)} frames")
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps
    cap.release()
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames, {video_duration:.2f} seconds")
    
    print("Generating audio...")
    audio = generate_audio(coordinates, width, height, fps, video_duration)
    
    print(f"Exporting audio to {output_path}")
    audio.export(output_path, format="wav")
    print(f"Done! Audio duration: {len(audio)}ms")

if __name__ == "__main__":
    main("bird_video.mp4", "bird_flight_synth.wav")