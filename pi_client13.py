#CS48007 Project Zeynep Uzun
import os
import time
import glob
import requests
import subprocess
import numpy as np
from PIL import Image
import io

# CONFIGURATION for our Rasp Pi
SERVER_IP = "192.168.1.104"    # this IP is the server IP with ipconfig
SERVER_PORT = 8000
POST_URL = f"http://{SERVER_IP}:{SERVER_PORT}/infer/frame"
BIN_GLOB = "frame-*.bin"
# lower resolution for faster throughput
RAW_W, RAW_H = 320, 240
RAW_BYTES = RAW_W * RAW_H * 3
JPEG_QUALITY = 70            # smaller payload meant our system worked faster
REQUEST_TIMEOUT = 5          # seconds
# only a few newest frames on disk are kept to avoid directory growth
KEEP_LAST_N_FRAMES = 2
# led action based on the frames
LED_BRIGHTNESS = "/sys/class/leds/led0/brightness"
LED_TRIGGER = "/sys/class/leds/led0/trigger"

def initLed():
    try:
        with open(LED_TRIGGER, "w") as f:
            f.write("none")
        with open(LED_BRIGHTNESS, "w") as f:
            f.write("0")
    except Exception:
        pass

def ledOn():
    try:
        with open(LED_BRIGHTNESS, "w") as f:
            f.write("1")
    except Exception:
        pass

def ledOff():
    try:
        with open(LED_BRIGHTNESS, "w") as f:
            f.write("0")
    except Exception:
        pass

def ledBlink(times, interval=0.12):
    for _ in range(times):
        ledOn()
        time.sleep(interval)
        ledOff()
        time.sleep(interval)

def cleanupFramesAll():
    for f in glob.glob(BIN_GLOB):
        try:
            os.remove(f)
        except Exception:
            pass

def cleanupKeepLatest(frames, keepN=KEEP_LAST_N_FRAMES):
    # deleting older .bin frames in the directory to only pass the new ones
    if not frames or len(frames) <= keepN:
        return
    try:
        # keeping the newest frames
        framesSorted = sorted(frames, key=os.path.getmtime)
        toDelete = framesSorted[:-keepN]
        for f in toDelete:
            try:
                os.remove(f)
            except Exception:
                pass
    except Exception:
        pass


def startCamera():
    subprocess.run(["pkill", "cam"], stderr=subprocess.DEVNULL)
    time.sleep(0.2)
    #our rasp pi did not have libcamera installed so we had to use the cam mcommand 
    cameraCmd = [
        "cam",
        "--camera", "/base/soc/i2c0mux/i2c@1/imx219@10",
        "--capture=0",
        "--file",
        f"--stream=pixelformat=RGB888,width={RAW_W},height={RAW_H}"
    ]
    return subprocess.Popen(cameraCmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def binToJpeg(binPath: str) -> bytes | None:
    try:
        with open(binPath, "rb") as f:
            rawBytes = f.read()
    except Exception:
        return None

    if len(rawBytes) != RAW_BYTES:
        return None
    imageArray = np.frombuffer(rawBytes, dtype=np.uint8).reshape((RAW_H, RAW_W, 3))
    image = Image.fromarray(imageArray, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buffer.getvalue()

def feedbackFromCommand(command: str, nearScore):
    if command == "clear":
        ledOff()
        return

    # higher score if the obstacle is near faster blink
    try:
        score = float(nearScore) if nearScore is not None else 0.25
    except Exception:
        score = 0.25

    if score >= 0.35:
        interval = 0.08
    elif score >= 0.25:
        interval = 0.11
    else:
        interval = 0.14

    if command == "obstacle_front":
        ledBlink(1, interval)
    elif command == "obstacle_left":
        ledBlink(2, interval)
    elif command == "obstacle_right":
        ledBlink(3, interval)
    else:
        ledBlink(1, 0.10)

def main():
    initLed()
    cleanupFramesAll()
    print(f"Pi client -> {POST_URL}")
    print("Press Ctrl+C to stop")
    camProc = startCamera()
    time.sleep(1.0)
    session = requests.Session()
    startTime = time.time()
    framesOk = 0
    framesError = 0
    # only keeps the last frame to send
    lastSeen = None

    try:
        while True:
            frames = glob.glob(BIN_GLOB)
            if not frames:
                time.sleep(0.01)
                continue

            # we pick the latest frame to send
            try:
                latest = max(frames, key=os.path.getmtime)
            except Exception:
                time.sleep(0.01)
                continue

            if latest == lastSeen:
                time.sleep(0.005)
                continue
            lastSeen = latest

            # cleans up the older frames to not get in the way of sending new ones
            cleanupKeepLatest(frames, KEEP_LAST_N_FRAMES)
            frameId = latest.split("-")[-1].split(".")[0]
            jpegBytes = binToJpeg(latest)
            if jpegBytes is None:
                continue
            uploadFiles = {"file": (f"{frameId}.jpg", jpegBytes, "image/jpeg")}
            data = {"frame_id": frameId, "client_capture_ts": str(time.time())}

            try:
                response = session.post(POST_URL, files=uploadFiles, data=data, timeout=REQUEST_TIMEOUT)
                if response.status_code != 200:
                    print(f"Server error {response.status_code}: {response.text[:120]}")
                    framesError += 1
                    continue
                result = response.json()
                command = result.get("command", "clear")
                score = result.get("near_score", None)
                endToEndLatency = float(result.get("end_to_end_latency_ms", 0))
                inferenceLatency = float(result.get("inference_latency_ms", 0))
                framesOk += 1
                scoreStr = f"{score:.3f}" if isinstance(score, (int, float)) else "None"
                print(f"{frameId} | {command:14s} score={scoreStr:6s} | e2e={endToEndLatency:.0f}ms infer={inferenceLatency:.0f}ms")
                feedbackFromCommand(command, score)
            except requests.exceptions.Timeout:
                print(f"Frame {frameId} timeout - skipping")
                framesError += 1
                ledOff()
            except Exception as error:
                print(f"Request error: {error}")
                framesError += 1
                ledOff()

    except KeyboardInterrupt:
        print("\n\nStopped by user (Ctrl+C)")
    finally:
        try:
            session.close()
        except Exception:
            pass

        camProc.terminate()
        cleanupFramesAll()
        ledOff()
        elapsed = time.time() - startTime
        print(f"\n{'='*50}")
        print("Session Summary:")
        print(f"Duration: {elapsed:.1f} seconds")
        print(f"Frames sent successfully: {framesOk}")
        print(f"Frames with errors: {framesError}")
        print(f"Average FPS: {framesOk/elapsed:.2f}")
        print(f"{'='*50}")

if __name__ == "__main__":
    main()
