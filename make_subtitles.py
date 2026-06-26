import sys
import os
from faster_whisper import WhisperModel

def format_time(seconds):
    """Formats seconds into the exact SRT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def transcribe_slovak_flawless(video_path):
    print(f"\n[1/3] Loading the massive 'large-v3' AI brain...")
    print("      (This might take a few minutes on older PCs, but accuracy is maximum)")
    
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")
    
    print(f"[2/3] Analyzing audio and transcribing flawless Slovak...")
    segments, info = model.transcribe(video_path, language="sk", word_timestamps=True)
    
    all_words = []
    for segment in segments:
        if segment.words:
            for word in segment.words:
                all_words.append(word)
                
    if not all_words:
        print("[-] Error: No speech detected in this video file.")
        return

    print(f"[3/3] Applying strict 3-word pacing rule...")
    output_srt_path = os.path.splitext(video_path)[0] + ".srt"
    
    counter = 1
    with open(output_srt_path, "w", encoding="utf-8") as srt_file:
        for i in range(0, len(all_words), 3):
            chunk = all_words[i:i+3]
            
            start_timestamp = format_time(chunk[0].start)
            end_timestamp = format_time(chunk[-1].end)
            text_line = " ".join([w.word.strip() for w in chunk])
            
            srt_file.write(f"{counter}\n")
            srt_file.write(f"{start_timestamp} --> {end_timestamp}\n")
            srt_file.write(f"{text_line}\n\n")
            counter += 1

    print(f"\n[SUCCESS] Flawless Slovak subtitles saved to:\n{output_srt_path}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[-] Error: Drag and drop a .mp4 file directly onto this tool.")
        input("\nPress Enter to exit...")
    else:
        try:
            transcribe_slovak_flawless(sys.argv[1])
        except Exception as e:
            print(f"[-] Critical Error: {str(e)}")
        finally:
            input("\nProcess finished. Press Enter to close...")
