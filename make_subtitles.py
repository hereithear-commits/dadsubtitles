import sys
import os
from faster_whisper import WhisperModel

def format_time(seconds):
    """Formats seconds into the exact SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def transcribe_slovak_bulletproof(video_path):
    print(f"\n[1/3] Initializing offline 'large-v3' model on CPU...")
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")
    
    print(f"[2/3] Filtering audio and transcribing Slovak text...")
    segments, info = model.transcribe(
        video_path, 
        language="sk", 
        word_timestamps=True,
        vad_filter=True,                                                
        vad_parameters=dict(min_silence_duration_ms=500, threshold=0.5), 
        condition_on_previous_text=False,                                
        beam_size=5,                                                     
        compression_ratio_threshold=2.4,                                 
        no_speech_threshold=0.6                                          
    )
    
    print(f"[3/3] Enforcing segment-aware 3-word pacing with pause detection...")
    output_srt_path = os.path.splitext(video_path)[0] + ".srt"
    
    counter = 1
    with open(output_srt_path, "w", encoding="utf-8") as srt_file:
        for segment in segments:
            if not hasattr(segment, 'words') or not segment.words:
                continue
            
            current_chunk = []
            for word in segment.words:
                if current_chunk:
                    # Calculate the silent gap between the last word and the current word
                    time_gap = word.start - current_chunk[-1].end
                    
                    # Force-break the subtitle if it hits 3 words OR if there is a natural pause > 0.8s
                    if len(current_chunk) >= 3 or time_gap > 0.8:
                        start_timestamp = format_time(current_chunk[0].start)
                        end_timestamp = format_time(current_chunk[-1].end)
                        text_line = " ".join([w.word.strip() for w in current_chunk])
                        
                        srt_file.write(f"{counter}\n")
                        srt_file.write(f"{start_timestamp} --> {end_timestamp}\n")
                        srt_file.write(f"{text_line}\n\n")
                        counter += 1
                        
                        current_chunk = []
                
                current_chunk.append(word)
            
            # Clear out any remaining words at the end of the segment
            if current_chunk:
                start_timestamp = format_time(current_chunk[0].start)
                end_timestamp = format_time(current_chunk[-1].end)
                text_line = " ".join([w.word.strip() for w in current_chunk])
                
                srt_file.write(f"{counter}\n")
                srt_file.write(f"{start_timestamp} --> {end_timestamp}\n")
                srt_file.write(f"{text_line}\n\n")
                counter += 1

    print(f"\n[SUCCESS] Flawless Slovak subtitles saved to:\n{output_srt_path}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[-] Error: Drag and drop a valid media file directly onto this script.")
        input("\nPress Enter to exit...")
    else:
        try:
            transcribe_slovak_bulletproof(sys.argv[1])
        except Exception as e:
            print(f"[-] Critical Runtime Error: {str(e)}")
        finally:
            input("\nProcess finished. Press Enter to close...")
