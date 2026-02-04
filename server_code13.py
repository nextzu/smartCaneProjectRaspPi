#Project CS48007 Zeynep Uzun 
import os
import time
import subprocess
from threading import Lock
import cv2
import numpy as np
from flask import Flask, request, jsonify, Response
from ultralytics import YOLO

app = Flask(__name__)

print("Loading YOLO model")
model = YOLO("yolov8n.pt")
print("Model loaded. Ready to receive frames.")

# COCO class names that we picked to use in our project that made sense in our in home and outside YOLOv8n training
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# streaming state
latestJpeg = None
latestLock = Lock()
# config for environment variables
SPEAK = os.getenv("SPEAK", "0") == "1"
SPEAK_COOLDOWN_SEC = float(os.getenv("SPEAK_COOLDOWN_SEC", "2.0"))  # cooldown for speak module 
SPEAK_SPECIFIC_OBJECTS = os.getenv("SPEAK_SPECIFIC_OBJECTS", "1") == "1"
SPEAK_THRESHOLD = 0.40  # reasonable threshold
OBSTACLE_MODE = os.getenv("OBSTACLE_MODE", "any")  # "any" or "person"
LEFT_THRESH = float(os.getenv("LEFT_THRESH", "0.33"))
RIGHT_THRESH = float(os.getenv("RIGHT_THRESH", "0.67"))
W_AREA = float(os.getenv("W_AREA", "0.7"))
W_BOTTOM = float(os.getenv("W_BOTTOM", "0.3"))
NEAR_IN_SCORE = float(os.getenv("NEAR_IN_SCORE", "0.20"))
NEAR_OUT_SCORE = float(os.getenv("NEAR_OUT_SCORE", "0.17"))
MIN_AREA_RATIO = float(os.getenv("MIN_AREA_RATIO", "0.01"))
# score smoothing to prevent jitter
SCORE_SMOOTH_FRAMES = 3
# state (anti-spam + hysteresis)
lastCommand = "clear"
lastObjectClass = None
lastNearScore = None
isNearState = False
lastSpokenAt = 0.0
lastSpokenCmd = None
lastSpokenObj = None
speechProcess = None
# score smoothing buffer
scoreHistory = []
speechProcess = None

def getClassName(clsId):
    #gets class names from COCO for us to understand
    if 0 <= clsId < len(COCO_CLASSES):
        return COCO_CLASSES[clsId]
    return f"object_{clsId}"

def getDistanceCategory(nearScore):
    #converts our near_score to interpretable distance levels
    if nearScore is None or nearScore < NEAR_OUT_SCORE:
        return "far"
    elif nearScore < 0.35:
        return "medium"
    elif nearScore < 0.50:
        return "close"
    else:
        return "very close"

def smoothNearScore(newScore):
    #smooths the near_score over multiple frames to reduce the changing when the frame is still
    global scoreHistory
    if newScore is None:
        scoreHistory = []
        return None
    scoreHistory.append(newScore)
    if len(scoreHistory) > SCORE_SMOOTH_FRAMES:
        scoreHistory.pop(0)
    return sum(scoreHistory) / len(scoreHistory)

def speak(command: str, objClassId: int = None):
    #announcing for obstacle warnings & only announces obstacles directly ahead to avoid confusion and waits for previous speech to finish before starting a new announcement 
    global lastSpokenAt, lastSpokenCmd, lastSpokenObj, speechProcess
    if not SPEAK or command == "clear":
        return
    # ONLY SPEAKS FOR OBSTACLES DIRECTLY AHEAD (not left/right)!!!!
    if command != "obstacle_front":
        return
    now = time.time()
    # waits for existing speech to finish instead of killing it
    if speechProcess is not None:
        if speechProcess.poll() is None:  # Still running
            return  # don't interrupt, just skip this announcement
    # checks if we should suppress due to cooldown
    sameCommand = (lastSpokenCmd == command and lastSpokenObj == objClassId)
    if sameCommand and (now - lastSpokenAt) < SPEAK_COOLDOWN_SEC:
        return
    # builds the text to speak
    if SPEAK_SPECIFIC_OBJECTS and objClassId is not None:
        objName = getClassName(objClassId)
        text = f"{objName} ahead"
    else:
        text = "Obstacle ahead"
    try:
        # since we used Windows in our project this part is made to be used with TTS using PowerShell
        psCommand = f'Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{text}")'
        speechProcess = subprocess.Popen(
            ["powershell", "-Command", psCommand], 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
        lastSpokenAt = now
        lastSpokenCmd = command
        lastSpokenObj = objClassId
        print(f"Speaking: '{text}'")
    except Exception as error:
        print("TTS error:", error)

def computeNearScore(x1, y1, x2, y2, frameW, frameH):
    # this function is a distance mettric we applied and the way it computes proximity score relies on two tjings:
    #70% weight means how much of frame the object fills (area_ratio)
    #30% weight means how close to bottom of frame (closer = nearer to camera)
    bboxWidth = float(x2 - x1)
    bboxHeight = float(y2 - y1)
    bboxArea = bboxWidth * bboxHeight
    frameArea = float(frameW * frameH)
    areaRatio = bboxArea / frameArea
    bottomRatio = float(y2) / float(frameH)
    score = W_AREA * areaRatio + W_BOTTOM * bottomRatio
    return score, areaRatio, bottomRatio

def applyNearHysteresis(rawCmd: str, score: float | None):
    # this function applies hysteresis to prevent rapid on/off toggling.
    # uses different thresholds for entering and exiting "near" state.
    global isNearState
    if score is None:
        isNearState = False
        return "clear"
    if not isNearState:
        # score needs higher score to trigger warning
        if score >= NEAR_IN_SCORE:
            isNearState = True
        else:
            return "clear"
    else:
        # score needs to drop below lower threshold to clear warning
        if score <= NEAR_OUT_SCORE:
            isNearState = False
            return "clear"
    return rawCmd

def pickObstaclesAndMain(results, frameW: int, frameH: int):
    # this function helps us process YOLO detections and pick the main obstacle
    allBoxes = []
    bestScore = -1.0
    bestArea = 0.0
    bestBox = None
    bestClsId = None
    bestDbg = None
    bestNormCx = None
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            clsId = int(box.cls)
            conf = float(box.conf) if box.conf is not None else None
            # filtering by obstacle mode
            if OBSTACLE_MODE == "person" and clsId != 0:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # clipping to frame boundaries
            x1 = max(0.0, min(float(frameW - 1), float(x1)))
            x2 = max(0.0, min(float(frameW - 1), float(x2)))
            y1 = max(0.0, min(float(frameH - 1), float(y1)))
            y2 = max(0.0, min(float(frameH - 1), float(y2)))

            if x2 <= x1 or y2 <= y1:
                continue
            score, areaRatio, bottomRatio = computeNearScore(x1, y1, x2, y2, frameW, frameH)
            # we filter out tiny detections to prevent noise
            if areaRatio < MIN_AREA_RATIO:
                continue
            bboxWidth = float(x2 - x1)
            bboxHeight = float(y2 - y1)
            area = bboxWidth * bboxHeight
            centerX = (x1 + x2) / 2.0
            normCx = centerX / float(frameW)
            item = {
                "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                "cls": clsId,
                "cls_name": getClassName(clsId),
                "conf": conf,
                "score": float(score),
                "area_ratio": float(areaRatio),
                "bottom_ratio": float(bottomRatio),
            }
            allBoxes.append(item)

            # we chhoose main obstacle by highest near_score and the tie-breaker is area
            if (score > bestScore) or (abs(score - bestScore) < 1e-6 and area > bestArea):
                bestScore = score
                bestArea = area
                bestBox = (float(x1), float(y1), float(x2), float(y2))
                bestClsId = clsId
                bestNormCx = normCx
                bestDbg = {
                    "cls_id": clsId,
                    "cls_name": getClassName(clsId),
                    "conf": conf,
                    "score": float(score),
                    "area_ratio": float(areaRatio),
                    "bottom_ratio": float(bottomRatio),
                    "norm_cx": float(normCx),
                    "bbox_w": float(bboxWidth),
                    "bbox_h": float(bboxHeight),
                }

    if bestBox is None:
        return "clear", None, None, None, allBoxes, None

    # we determine direction based on horizontal position based on 0.33 and 0.67 thresholds
    if bestNormCx < LEFT_THRESH:
        rawCmd = "obstacle_left"
    elif bestNormCx > RIGHT_THRESH:
        rawCmd = "obstacle_right"
    else:
        rawCmd = "obstacle_front"
    return rawCmd, float(bestScore), bestBox, bestClsId, allBoxes, bestDbg

def drawOverlay(image, allBoxes, bestBox, command, score, inferMs, e2eMs, bestClsId=None):
    # this function helps us draw detection overlay on image for visualization 
    overlay = image.copy()
    # draws all detections in yellow
    for boxData in allBoxes:
        x1, y1, x2, y2 = int(boxData["x1"]), int(boxData["y1"]), int(boxData["x2"]), int(boxData["y2"])
        clsName = boxData["cls_name"]
        boxScore = boxData["score"]
        conf = boxData["conf"]
        label = f"{clsName} s={boxScore:.2f}"
        if conf is not None:
            label += f" p={conf:.2f}"
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(overlay, label, (x1, max(20, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    # highlighting main obstacle in red
    if bestBox is not None:
        x1, y1, x2, y2 = map(int, bestBox)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)
        # adding label for main obstacle
        if bestClsId is not None:
            mainLabel = f"MAIN: {getClassName(bestClsId)}"
            cv2.putText(overlay, mainLabel, (x1, max(40, y1 - 25)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # draws status text
    distanceCat = getDistanceCategory(score)
    text1 = f"cmd: {command}"
    text2 = f"near_score: {score:.3f} ({distanceCat})" if score is not None else "near_score: None"
    text3 = f"infer: {inferMs:.1f}ms  e2e: {e2eMs:.1f}ms"
    cv2.putText(overlay, text1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(overlay, text2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(overlay, text3, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    return overlay

@app.route("/infer/frame", methods=["POST"])
def inferFrame():
    # this is the main inference endpoint - receives frame, returns obstacle detection result
    global lastCommand, lastObjectClass, lastNearScore, latestJpeg
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    frameId = request.form.get("frame_id", "unknown")
    clientTs = request.form.get("client_capture_ts")
    if clientTs:
        try:
            clientTs = float(clientTs)
        except Exception:
            clientTs = time.time()
    else:
        clientTs = time.time()

    # decoding image
    fileBytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(fileBytes, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Could not decode image"}), 400

    frameH, frameW = image.shape[:2]
    # running YOLO inference
    startTime = time.time()
    results = model(image, verbose=False)
    endTime = time.time()
    # processing detections
    rawCmd, rawScore, bestBox, bestClsId, allBoxes, bestDbg = pickObstaclesAndMain(results, frameW, frameH)
    # apply score smoothing to reduce jitter
    score = smoothNearScore(rawScore)
    # apply hysteresis
    gatedCmd = applyNearHysteresis(rawCmd, score)

    # we have to decide if we should speak
    # only speaks if the obstacle is ahead, score above threshold and state changed
    shouldSpeak = (
        gatedCmd == "obstacle_front" and  # only speak for front obstacles
        score is not None and 
        score > SPEAK_THRESHOLD and 
        (
            (lastCommand == "clear") or 
            (gatedCmd != lastCommand) or 
            (bestClsId != lastObjectClass)
        )
    )

    inferenceLatency = (endTime - startTime) * 1000.0
    endToEndLatency = (time.time() - clientTs) * 1000.0
    # update states
    lastCommand = gatedCmd
    lastObjectClass = bestClsId if gatedCmd != "clear" else None
    lastNearScore = score if gatedCmd != "clear" else None
    # trigger speech if needed
    if shouldSpeak:
        speak(gatedCmd, bestClsId)
    # create overlay and store for streaming
    overlay = drawOverlay(image, allBoxes, bestBox, gatedCmd, score, inferenceLatency, endToEndLatency, bestClsId)
    success, jpeg = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if success:
        with latestLock:
            latestJpeg = jpeg.tobytes()
    objName = getClassName(bestClsId) if bestClsId is not None else "none"
    distanceCat = getDistanceCategory(score)
    scoreStr = f"{score:.3f}" if score is not None else "None"
    print(f"Frame {frameId} | raw={rawCmd} obj={objName} score={scoreStr} ({distanceCat}) gated={gatedCmd} | infer={inferenceLatency:.1f}ms e2e={endToEndLatency:.1f}ms")
    return jsonify({
        "frame_id": frameId,
        "command": gatedCmd,
        "object_class": objName if gatedCmd != "clear" else None,
        "near_score": lastNearScore,
        "distance_category": distanceCat,
        "inference_latency_ms": inferenceLatency,
        "end_to_end_latency_ms": endToEndLatency,
        "debug": bestDbg
    })

@app.route("/frame.jpg")
def frameJpg():
    # only gets latest processed frame as JPEG
    with latestLock:
        if latestJpeg is None:
            return "No frame yet", 404
        data = latestJpeg
    return Response(data, mimetype="image/jpeg")

@app.route("/stream")
def stream():
    # MJPEG stream of processed frames
    def gen():
        boundary = b"--frame"
        while True:
            with latestLock:
                frame = latestJpeg
            if frame is None:
                time.sleep(0.03)
                continue

            yield boundary + b"\r\n"
            yield b"Content-Type: image/jpeg\r\n"
            yield b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
            yield frame + b"\r\n"
            time.sleep(0.03)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status", methods=["GET"])
def status():
    # gets the current system status
    return jsonify({
        "last_command": lastCommand,
        "last_object_class": getClassName(lastObjectClass) if lastObjectClass is not None else None,
        "last_near_score": lastNearScore,
        "distance_category": getDistanceCategory(lastNearScore),
        "near_in_score": NEAR_IN_SCORE,
        "near_out_score": NEAR_OUT_SCORE,
        "speak_threshold": SPEAK_THRESHOLD,
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)