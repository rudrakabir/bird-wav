import cv2
import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine

def extract_bird_coordinates(video_path):
    cap = cv2.VideoCapture(video_path)
    backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    bird_coordinates = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        fgMask = backSub.apply(frame)
        _, thresh = cv2.threshold(fgMask, 244, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        frame_coordinates = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Adjust this threshold as needed
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    frame_coordinates.append((cX, cY))
        
        bird_coordinates.append(frame_coordinates)
    
    cap.release()
    return bird_coordinates

def coordinates_to_frequency(x, y, width, height):
    min_freq, max_freq = 220, 880
    freq = min_freq + (x / width) * (max_freq - min_freq)
    return freq

def generate_audio(coordinates, width, height, fps=30):
    audio = AudioSegment.silent(duration=0)
    frame_duration = 1000 // fps  # Duration of each frame in milliseconds
    crossfade_duration = min(frame_duration // 2, 50)  # Crossfade between frames, max 50ms
    
    for frame_num, frame_coords in enumerate(coordinates):
        frame_audio = AudioSegment.silent(duration=frame_duration)
        print(f"Processing frame {frame_num+1}/{len(coordinates)}, {len(frame_coords)} birds detected")
        
        for x, y in frame_coords:
            freq = coordinates_to_frequency(x, y, width, height)
            print(f"  Bird at ({x}, {y}): freq={freq:.2f}Hz")
            tone = Sine(freq).to_audio_segment(duration=frame_duration)
            # Apply low-pass filter for a softer sound
            tone = low_pass_filter(tone, 1000)
            # Fade in and out for smoother transitions
            tone = tone.fade_in(crossfade_duration).fade_out(crossfade_duration)
            frame_audio = frame_audio.overlay(tone, gain_during_overlay=-6)
        
        if len(audio) > 0:
            audio = audio.append(frame_audio, crossfade=crossfade_duration)
        else:
            audio += frame_audio
        
        print(f"  Total audio duration: {len(audio)}ms")
    
    # Add a gentle background ambience
    bg_noise = AudioSegment.silent(duration=len(audio)).overlay(
        Sine(80).to_audio_segment(duration=len(audio)), gain_during_overlay=-20
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
    cap.release()
    
    print(f"Video dimensions: {width}x{height}, FPS: {fps}")
    
    print("Generating audio...")
    audio = generate_audio(coordinates, width, height, fps)
    
    print(f"Exporting audio to {output_path}")
    audio.export(output_path, format="wav")
    print(f"Done! Audio duration: {len(audio)}ms")

if __name__ == "__main__":
    main("bird_video.mp4", "bird_flight_synth.wav")