import sys
import os
from faster_whisper import WhisperModel

def format_time(seconds):
    """Formats seconds into the exact SRT timestamp format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def transcribe_slovak_fast(video_path):
    print(f"\n[1/3] Loading offline Slovak AI model (CPU Mode)...")
    # Using 'base' for a perfect balance of fast speed and high accuracy on standard laptops
    model = WhisperModel("base", device="cpu", compute_type="int8")
    
    print(f"[2/3] Analyzing audio and transcribing language: Slovak...")
    segments, info = model.transcribe(video_path, language="sk", word_timestamps=True)
    
    # Collect every single word with its individual timestamp
    all_words = []
    for segment in segments:
        if segment.words:
            for word in segment.words:
                all_words.append(word)
                
    if not all_words:
        print("[-] Error: No speech detected in this video file.")
        return

    print(f"[3/3] Applying strict 3-word pacing rule and formatting...")
    output_srt_path = os.path.splitext(video_path)[0] + ".srt"
    
    counter = 1
    with open(output_srt_path, "w", encoding="utf-8") as srt_file:
        # Loop through all words in strict increments of 3
        for i in range(0, len(all_words), 3):
            chunk = all_words[i:i+3]
            
            start_timestamp = format_time(chunk[0].start)
            end_timestamp = format_time(chunk[-1].end)
            text_line = " ".join([w.word.strip() for w in chunk])
            
            # Write standard SRT block structure
            srt_file.write(f"{counter}\n")
            srt_file.write(f"{start_timestamp} --> {end_timestamp}\n")
            srt_file.write(f"{text_line}\n\n")
            counter += 1

    print(f"[SUCCESS] Saved perfect Slovak subtitles to:\n{output_srt_path}\n")

if __name__ == "__main__":
    # Check if a file was actually dragged and dropped onto the application
    if len(sys.argv) < 2:
        print("[-] Error: Drag and drop a widescreen .mp4 file directly onto this tool.")
        input("\nPress Enter to exit...")
    else:
        target_video = sys.argv[1]
        try:
            transcribe_slovak_fast(target_video)
        except Exception as e:
            print(f"[-] Critical Error: {str(e)}")
        finally:
            input("\nProcess finished. Press Enter to close...")
