"""
Object detection demo.

This demo script requires Raspberry Pi Camera, and pre-compiled mode.
Get pre-compiled model from Coral website.
$ wget https://dl.google.com/coral/canned_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite
"""

from edgetpu.detection.engine import DetectionEngine
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np
import time
import io
import picamera


# https://github.com/waveform80/picamera/issues/383
def _monkey_patch_picamera():
    original_send_buffer = picamera.mmalobj.MMALPortPool.send_buffer

    def silent_send_buffer(zelf, *args, **kwargs):
        try:
            original_send_buffer(zelf, *args, **kwargs)
        except picamera.exc.PiCameraMMALError as error:
            if error.status != 14:
                raise error

    picamera.mmalobj.MMALPortPool.send_buffer = silent_send_buffer


# Read labels.txt file provided by Coral website
def _read_label_file(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    ret = {}
    for line in lines:
        pair = line.strip().split(maxsplit=1)
        ret[int(pair[0])] = pair[1].strip()
    return ret


# Main loop
def main():
    model_filename = "mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"
    label_filename = "coco_labels.txt"
    engine = DetectionEngine(model_filename)
    labels = _read_label_file(label_filename)
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480

    fnt = ImageFont.load_default()

    # To view preview on VNC,
    # https://raspberrypi.stackexchange.com/a/74390
    with picamera.PiCamera() as camera:
        _monkey_patch_picamera()
        camera.resolution = (CAMERA_WIDTH, CAMERA_HEIGHT)
        camera.framerate = 15
        camera.rotation = 180
        _, width, height, channels = engine.get_input_tensor_shape()
        print("{}, {}".format(width, height))
        overlay_renderer = None
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
                input = input.reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 3))
                image = Image.fromarray(input)
                # image.save("out.jpg")

                # Make overlay image plane
                img = Image.new('RGBA',
                                (CAMERA_WIDTH, CAMERA_HEIGHT),
                                (255, 0, 0, 0))
                draw = ImageDraw.Draw(img)

                # Run detection
                start_ms = time.time()
                results = engine.DetectWithImage(image,
                                                 threshold=0.2, top_k=10)
                elapsed_ms = (time.time() - start_ms)*1000.0
                if results:
                    for obj in results:
                        box = obj.bounding_box.flatten().tolist()
                        box[0] *= CAMERA_WIDTH
                        box[1] *= CAMERA_HEIGHT
                        box[2] *= CAMERA_WIDTH
                        box[3] *= CAMERA_HEIGHT
                        # print(box)
                        # print(labels[obj.label_id])
                        draw.rectangle(box, outline='red')
                        draw.text((box[0], box[1]-10), labels[obj.label_id],
                                  font=fnt, fill="red")
                    camera.annotate_text = "{0:.2f}ms".format(elapsed_ms)
                if not overlay_renderer:
                    overlay_renderer = camera.add_overlay(
                        img.tobytes(),
                        size=(CAMERA_WIDTH, CAMERA_HEIGHT), layer=4, alpha=255)
                else:
                    overlay_renderer.update(img.tobytes())
        finally:
            if overlay_renderer:
                camera.remove_overlay(overlay_renderer)
            camera.stop_preview()


if __name__ == "__main__":
    main()
