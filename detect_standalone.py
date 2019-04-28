

from edgetpu.detection.engine import DetectionEngine
from PIL import Image
from PIL import ImageDraw

import numpy as np
import time
import io
import picamera

model_filename = "mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"
engine = DetectionEngine(model_filename)

with picamera.PiCamera() as camera:
    camera.resolution = (640, 480)
    camera.framerate = 30
    _, width, height, channels = engine.get_input_tensor_shape()
    camera.start_preview()
    try:
        stream = io.BytesIO()
        for foo in camera.capture_continuous(stream,
                                             format='rgb',
                                             use_video_port=True,
                                             resize=(width, height)):
            stream.truncate()
            stream.seek(0)
            input = np.frombuffer(stream.getvalue(), dtype=np.uint8)
            start_ms = time.time()
            results = engine.DetectWithImage(input, threshold=0.1, top_k=3)
            elapsed_ms = time.time() - start_ms
            if results:
                print(results)
                # camera.annotate_text = "%s %.2f\n%.2fms" % (
                #    labels[results[0][0]], results[0][1], elapsed_ms*1000.0)
    finally:
        camera.stop_preview()
