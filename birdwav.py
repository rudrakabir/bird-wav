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
    min_duration, max_duration = 50, 200
    freq = min_freq + (x / width) * (max_freq - min_freq)
    duration = min_duration + (y / height) * (max_duration - min_duration)
    return freq, duration

def generate_audio(coordinates, width, height):
    audio = AudioSegment.silent(duration=0)
    for frame_num, frame_coords in enumerate(coordinates):
        frame_audio = AudioSegment.silent(duration=0)
        print(f"Processing frame {frame_num+1}/{len(coordinates)}, {len(frame_coords)} birds detected")
        for x, y in frame_coords:
            freq, duration = coordinates_to_frequency(x, y, width, height)
            print(f"  Bird at ({x}, {y}): freq={freq:.2f}Hz, duration={duration:.2f}ms")
            tone = Sine(freq).to_audio_segment(duration=duration)
            frame_audio = frame_audio.overlay(tone)
        audio += frame_audio
    
    print(f"Final audio duration: {len(audio)}ms")
    return audio

def main(video_path, output_path):
    print("Extracting bird coordinates...")
    coordinates = extract_bird_coordinates(video_path)
    print(f"Extracted coordinates from {len(coordinates)} frames")
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"Video dimensions: {width}x{height}")
    
    print("Generating audio...")
    audio = generate_audio(coordinates, width, height)
    
    print(f"Exporting audio to {output_path}")
    audio.export(output_path, format="wav")
    print(f"Done! Audio duration: {len(audio)}ms")

if __name__ == "__main__":
    main("bird_video.mp4", "bird_flight_synth.wav")