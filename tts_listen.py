#!/usr/bin/env python3
"""Speak whatever you type, 2 seconds after you stop typing."""
import argparse
import queue
import select
import sys
import termios
import threading
import time

import sounddevice as sd
import torch
from qwen_tts import Qwen3TTSModel

IDLE_SECONDS = 2.0
MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


def speak_worker(model, language, speaker, speak_queue):
    while True:
        text = speak_queue.get()
        if text is None:
            break
        wavs, sr = model.generate_custom_voice(text=text, language=language, speaker=speaker)
        sd.play(wavs[0], sr)
        sd.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speaker", default="Ryan")
    parser.add_argument("--language", default="English")
    args = parser.parse_args()

    print(f"Loading {MODEL_NAME} ...", file=sys.stderr)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Qwen3TTSModel.from_pretrained(MODEL_NAME, device_map=device, dtype=torch.bfloat16)
    print("Ready. Start typing -- speech plays 2s after you stop. Ctrl+C to quit.", file=sys.stderr)

    speak_queue = queue.Queue()
    worker = threading.Thread(
        target=speak_worker, args=(model, args.language, args.speaker, speak_queue), daemon=True
    )
    worker.start()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    new_settings = termios.tcgetattr(fd)
    new_settings[3] = new_settings[3] & ~(termios.ICANON | termios.ECHO)
    termios.tcsetattr(fd, termios.TCSANOW, new_settings)

    buffer = []
    last_keystroke = None

    try:
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready:
                ch = sys.stdin.read(1)
                if ch in ("\x7f", "\x08"):  # backspace
                    if buffer:
                        buffer.pop()
                        sys.stdout.write("\b \b")
                        sys.stdout.flush()
                elif ch in ("\r", "\n"):
                    buffer.append(" ")
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                else:
                    buffer.append(ch)
                    sys.stdout.write(ch)
                    sys.stdout.flush()
                last_keystroke = time.monotonic()
            elif last_keystroke is not None and buffer and (time.monotonic() - last_keystroke) >= IDLE_SECONDS:
                text = "".join(buffer).strip()
                buffer.clear()
                last_keystroke = None
                if text:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    speak_queue.put(text)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        speak_queue.put(None)
        print("\nExiting.")


if __name__ == "__main__":
    main()
