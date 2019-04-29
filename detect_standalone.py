

from edgetpu.detection.engine import DetectionEngine
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import numpy as np
import time
import io
import picamera

def ReadLabelFile(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    ret = {}
    for line in lines:
        pair = line.strip().split(maxsplit=1)
        ret[int(pair[0])] = pair[1].strip()
    return ret


model_filename = "mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"
label_filename = "coco_labels.txt"
engine = DetectionEngine(model_filename)
labels = ReadLabelFile(label_filename)

# To view preview on VNC,
# https://raspberrypi.stackexchange.com/a/74390

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

fnt = ImageFont.load_default()

with picamera.PiCamera() as camera:
    camera.resolution = (CAMERA_WIDTH, CAMERA_HEIGHT)
    camera.framerate = 30
    camera.rotation = 180
    _, width, height, channels = engine.get_input_tensor_shape()
    print("{}, {}".format(width, height))
    o = None
    camera.start_preview()
    try:
        stream = io.BytesIO()
        for foo in camera.capture_continuous(stream,
                                             format='rgb',
                                             use_video_port=True):
            # Make Image object from camera stream
            stream.truncate()
            stream.seek(0)
            input = np.frombuffer(stream.getvalue(), dtype=np.uint8)
            image = Image.fromarray(input.reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 3)))
            # image.save("out.jpg")

            # Make overlay image plane
            img = Image.new('RGBA', (CAMERA_WIDTH, CAMERA_HEIGHT), (255, 0, 0, 0))
            draw = ImageDraw.Draw(img)

            # Run detection
            start_ms = time.time()
            results = engine.DetectWithImage(image, threshold=0.2, top_k=10)
            elapsed_ms = time.time() - start_ms

            if results:
                for obj in results:
                    box = obj.bounding_box.flatten().tolist()
                    box[0] *= CAMERA_WIDTH
                    box[1] *= CAMERA_HEIGHT
                    box[2] *= CAMERA_WIDTH
                    box[3] *= CAMERA_HEIGHT
                    print(box)
                    print(labels[obj.label_id])
                    draw.rectangle(box, outline='red')
                    draw.text((box[0], box[1]-10), labels[obj.label_id], font=fnt, fill="red") 
                if not o:
                    o = camera.add_overlay(img.tobytes(), size=(CAMERA_WIDTH, CAMERA_HEIGHT), layer=3, alpha=255)
                else:
                    o.update(img.tobytes())
                camera.annotate_text = "{0:.2f}ms".format(elapsed_ms*1000.0)
    finally:
        if o:
            camera.remove_overlay(o)
        camera.stop_preview()

