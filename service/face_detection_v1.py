
from __future__ import print_function
import sys
import os
import cv2
import logging as log
from openvino.inference_engine import IENetwork, IEPlugin
from pathlib import Path


class FaceDetectionService:

    feature = 'Face Detection'
    model_xml_path = str(Path(__file__).resolve().parent.parent.joinpath(
        "model/intel/face-detection-retail-0004/FP32/face-detection-retail-0004.xml"))
    cpu_extension = str(Path(__file__).resolve().parent.parent.joinpath("model/lib/libcpu_extension.so"))
    prob_threshold = 0.5

    def __init__(self, device="CPU"):
        self.device = device
        self.model_bin = os.path.splitext(self.model_xml_path)[0] + ".bin"
        self.out_blob = None
        self.input_blob = None
        self.exec_net = None
        self.n, self.c, self.h, self.w = None, None, None, None
        self.input_stream = 0
        self.cur_request_id = 0
        self.frames_counter = 0
        self.plugin = None
        self.cap = None
        self.__get_video_capture()

    def load_model(self):
        log.info("Initializing plugin for {} device...".format(self.device))
        plugin = IEPlugin(self.device, None)

        log.info("Loading network files for {}".format(self.feature))
        plugin.add_cpu_extension(self.cpu_extension)
        net = IENetwork(model=self.model_xml_path, weights=self.model_bin)

        log.info("Checking {} network inputs".format(self.feature))
        return plugin, net

    def detection(self):
        # Make sure only one IEPlugin was created for one type of device
        self.plugin, net = self.load_model()

        # Face detection
        self.input_blob = next(iter(net.inputs))
        self.out_blob = next(iter(net.outputs))
        self.exec_net = self.plugin.load(network=net, num_requests=2)
        self.n, self.c, self.h, self.w = net.inputs[self.input_blob].shape
        del net

    def __get_video_capture(self):
        cap = cv2.VideoCapture(self.input_stream)
        if not cap.isOpened():
            sys.exit(1)
        return cap

    def processing_frame(self):
        cap = self.__get_video_capture()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self.frames_counter += 1
            initial_w = cap.get(3)
            initial_h = cap.get(4)

            in_frame = cv2.resize(frame, (self.w, self.h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((self.n, self.c, self.h, self.w))
            self.exec_net.start_async(request_id=self.cur_request_id, inputs={self.input_blob: in_frame})
            if self.exec_net.requests[self.cur_request_id].wait(-1) == 0:

                # Parse detection results of the current request
                res = self.exec_net.requests[self.cur_request_id].outputs[self.out_blob]
                for obj in res[0][0]:
                    # Draw only objects when probability more than specified threshold
                    if obj[2] > self.prob_threshold:
                        xmin = int(obj[3] * initial_w)
                        ymin = int(obj[4] * initial_h)
                        xmax = int(obj[5] * initial_w)
                        ymax = int(obj[6] * initial_h)

                        class_id = int(obj[1])
                        # Draw box and label\class_id
                        color = (min(class_id * 12.5, 255), min(class_id * 7, 255), min(class_id * 5, 255))
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

                        cv2.imwrite('controller/static/images/frame.jpg', frame)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + open('controller/static/images/frame.jpg',
                                                                          'rb').read() + b'\r\n')

        del self.exec_net
        del self.plugin
        log.info("Execution successful")
