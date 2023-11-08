import os
import pandas as pd
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
#?
from shapely.geometry import LineString, Polygon, Point

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
flags.DEFINE_boolean('fourcounter',False, 'count objects passes through the line')
flags.DEFINE_boolean('fivecounter',False, 'count objects passes through the line')
flags.DEFINE_boolean('sixcounter',False, 'count objects passes through the line')

def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0

    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
    
    # get ready for fourcounter flag
    if FLAGS.fourcounter:
        list_overlapping_leg1out = []
        list_overlapping_leg2out = []
        list_overlapping_leg3out = []
        list_overlapping_leg4out = []
        l1_to_l2_count = 0
        l2_to_l1_count = 0
        l1_to_l1_count = 0
        l1_to_l3_count = 0
        l1_to_l4_count = 0
        l2_to_l2_count = 0
        l2_to_l3_count = 0
        l2_to_l4_count = 0
        l3_to_l1_count = 0
        l3_to_l2_count = 0
        l3_to_l3_count = 0
        l3_to_l4_count = 0
        l4_to_l1_count = 0
        l4_to_l2_count = 0
        l4_to_l3_count = 0
        l4_to_l4_count = 0

        class_counter_1enter1 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_1enter2 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_1enter3 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_1enter4 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_2enter1 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_2enter2 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_2enter3 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_2enter4 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_3enter1 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_3enter2 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_3enter3 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_3enter4 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_4enter1 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_4enter2 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_4enter3 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_4enter4 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
    

    # get ready for fivecounter flag
    if FLAGS.fivecounter:
        list_overlapping_leg1out = []
        list_overlapping_leg2out = []
        list_overlapping_leg3out = []
        list_overlapping_leg4out = []
        list_overlapping_leg5out = []
        l1_to_l1_count = 0
        l1_to_l2_count = 0
        l1_to_l3_count = 0
        l1_to_l4_count = 0
        l1_to_l5_count = 0
        l2_to_l1_count = 0
        l2_to_l2_count = 0
        l2_to_l3_count = 0
        l2_to_l4_count = 0
        l2_to_l5_count = 0
        l3_to_l1_count = 0
        l3_to_l2_count = 0
        l3_to_l3_count = 0
        l3_to_l4_count = 0
        l3_to_l5_count = 0
        l4_to_l1_count = 0
        l4_to_l2_count = 0
        l4_to_l3_count = 0
        l4_to_l4_count = 0
        l4_to_l5_count = 0
        l5_to_l1_count = 0
        l5_to_l2_count = 0
        l5_to_l3_count = 0
        l5_to_l4_count = 0
        l5_to_l5_count = 0

        class_counter_1enter1 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_1enter2 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_1enter3 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_1enter4 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_1enter5 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_2enter1 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_2enter2 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_2enter3 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_2enter4 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_2enter5 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_3enter1 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_3enter2 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_3enter3 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_3enter4 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_3enter5 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_4enter1 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_4enter2 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_4enter3 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_4enter4 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_4enter5 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_5enter1 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_5enter2 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_5enter3 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_5enter4 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_5enter5 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}

    # get ready for sixcounter flag
    if FLAGS.sixcounter:
        list_overlapping_leg1out = []
        list_overlapping_leg2out = []
        list_overlapping_leg3out = []
        list_overlapping_leg4out = []
        list_overlapping_leg5out = []
        list_overlapping_leg6out = []
        l1_to_l1_count = 0
        l1_to_l2_count = 0
        l1_to_l3_count = 0
        l1_to_l4_count = 0
        l1_to_l5_count = 0
        l1_to_l6_count = 0
        l2_to_l1_count = 0
        l2_to_l2_count = 0
        l2_to_l3_count = 0
        l2_to_l4_count = 0
        l2_to_l5_count = 0
        l2_to_l6_count = 0
        l3_to_l1_count = 0
        l3_to_l2_count = 0
        l3_to_l3_count = 0
        l3_to_l4_count = 0
        l3_to_l5_count = 0
        l3_to_l6_count = 0
        l4_to_l1_count = 0
        l4_to_l2_count = 0
        l4_to_l3_count = 0
        l4_to_l4_count = 0
        l4_to_l5_count = 0
        l4_to_l6_count = 0
        l5_to_l1_count = 0
        l5_to_l2_count = 0
        l5_to_l3_count = 0
        l5_to_l4_count = 0
        l5_to_l5_count = 0
        l5_to_l6_count = 0
        l6_to_l1_count = 0
        l6_to_l2_count = 0
        l6_to_l3_count = 0
        l6_to_l4_count = 0
        l6_to_l5_count = 0
        l6_to_l6_count = 0

        class_counter_1enter1 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_1enter2 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_1enter3 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_1enter4 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_1enter5 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_1enter6 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_2enter1 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_2enter2 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_2enter3 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_2enter4 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_2enter5 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_2enter6 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_3enter1 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_3enter2 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_3enter3 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_3enter4 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_3enter5 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_3enter6 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_4enter1 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_4enter2 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_4enter3 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_4enter4 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_4enter5 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_4enter6 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_5enter1 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_5enter2 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_5enter3 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_5enter4 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_5enter5 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_5enter6 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_6enter1 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_6enter2 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_6enter3 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_6enter4 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_6enter5 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}
        class_counter_6enter6 = {'car': 0,'van': 0, 'small-truck': 0, 'large-truck': 0,'bus': 0,'motorbike': 0}

        last_frame = None



    frame_num = 0
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())

        # custom allowed classes (uncomment line below to customize tracker for only people)
        #allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]


        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()

            #draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            # draw circle at the center of the bbox
            x_center = int((bbox[0] + bbox[2]) / 2)
            y_center = int((bbox[1] + bbox[3]) / 2)
            radius = 5
            # draw bbox and circle at centre
            cv2.circle(frame, (x_center, y_center), radius, color, -1)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

            
            #? If enable fourcounter flag
            if FLAGS.fourcounter:
                #Blue Line
                x1, y1 = 1310,960
                x2, y2 = 885,1030
                x3, y3 = 400,950
                cv2.line(frame, (x1, y1), (x2,y2), (0, 0, 255), 10)
                cv2.line(frame, (x2, y2), (x3,y3), (0, 0, 200), 10)
                cv2.putText(frame,('Leg 1'),(x1+5,y1+5), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,0,255),2)
                            
                #Yellow Line
                x4, y4 = 270, 750
                x5, y5 = 230, 420
                x6, y6 = 420, 200
                cv2.line(frame, (x4, y4), (x5, y5), (255, 255, 0), 10)
                cv2.line(frame, (x5, y5), (x6, y6), (255, 200, 0), 10)
                cv2.putText(frame,('Leg 2'),(x4+5,y4+5), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(255,255,0),2)

                #Red Line
                x7, y7 = 600,85
                x8, y8 = 930,40
                x9, y9 = 1350,100
                cv2.line(frame, (x7, y7), (x8, y8), (255, 0, 0), 10)
                cv2.line(frame, (x8, y8), (x9, y9), (200, 0, 0), 10)
                cv2.putText(frame,('Leg 3'),(x7+20,y7+20), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(255,0,0),2)

                #Green Line
                x10, y10 = 1430,145
                x11, y11 = 1600,400
                x12, y12 = 1580,750
                cv2.line(frame, (x10, y10), (x11, y11), (0, 255, 0), 10)
                cv2.line(frame, (x11, y11), (x12, y12), (0, 200, 0), 10)
                cv2.putText(frame,('Leg 4'),(x10+10,y10+5), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,255,0),2)

                # Define the line segments
                leg1in = LineString([(x1, y1), (x2, y2)])
                leg1out = LineString([(x2, y2), (x3, y3)])
                leg2in = LineString([(x4, y4), (x5, y5)])
                leg2out = LineString([(x5, y5), (x6, y6)])
                leg3in = LineString([(x7, y7), (x8, y8)])
                leg3out = LineString([(x8, y8), (x9, y9)])
                leg4in = LineString([(x10, y10), (x11, y11)])
                leg4out = LineString([(x11, y11), (x12, y12)])

                # Define the circle
                circle = Point(x_center, y_center).buffer(radius)

                if leg1out.intersects(circle):
                    if track.track_id not in list_overlapping_leg1out:
                        list_overlapping_leg1out.append(track.track_id)
                    pass

                if leg2out.intersects(circle):
                    if track.track_id not in list_overlapping_leg2out:
                        list_overlapping_leg2out.append(track.track_id)
                    pass

                if leg3out.intersects(circle):
                    if track.track_id not in list_overlapping_leg3out:
                        list_overlapping_leg3out.append(track.track_id)
                    pass

                if leg4out.intersects(circle):
                    if track.track_id not in list_overlapping_leg4out:
                        list_overlapping_leg4out.append(track.track_id)
                    pass

                if leg1in.intersects(circle):
                    if track.track_id in list_overlapping_leg1out:
                        l1_to_l1_count += 1
                        list_overlapping_leg1out.remove(track.track_id)
                        if class_name not in class_counter_1enter1:
                            class_counter_1enter1[class_name] = 1
                        else:
                            class_counter_1enter1[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg2out:
                        l2_to_l1_count += 1
                        list_overlapping_leg2out.remove(track.track_id)
                        if class_name not in class_counter_2enter1:
                            class_counter_2enter1[class_name] = 1
                        else:
                            class_counter_2enter1[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg3out:
                        l3_to_l1_count += 1
                        list_overlapping_leg3out.remove(track.track_id)
                        if class_name not in class_counter_3enter1:
                            class_counter_3enter1[class_name] = 1
                        else:
                            class_counter_3enter1[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg4out:
                        l4_to_l1_count += 1
                        list_overlapping_leg4out.remove(track.track_id)
                        if class_name not in class_counter_4enter1:
                            class_counter_4enter1[class_name] = 1
                        else:
                            class_counter_4enter1[class_name] += 1
                            pass
                        pass

                    else:
                        pass

                if leg2in.intersects(circle):
                    if track.track_id in list_overlapping_leg1out:
                        l1_to_l2_count += 1
                        list_overlapping_leg1out.remove(track.track_id)
                        if class_name not in class_counter_1enter2:
                            class_counter_1enter2[class_name] = 1
                        else:
                            class_counter_1enter2[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg2out:
                        l2_to_l2_count += 1
                        list_overlapping_leg2out.remove(track.track_id)
                        if class_name not in class_counter_2enter2:
                            class_counter_2enter2[class_name] = 1
                        else:
                            class_counter_2enter2[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg3out:
                        l3_to_l2_count += 1
                        list_overlapping_leg3out.remove(track.track_id)
                        if class_name not in class_counter_3enter2:
                            class_counter_3enter2[class_name] = 1
                        else:
                            class_counter_3enter2[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg4out:
                        l4_to_l2_count += 1
                        list_overlapping_leg4out.remove(track.track_id)
                        if class_name not in class_counter_4enter2:
                            class_counter_4enter2[class_name] = 1
                        else:
                            class_counter_4enter2[class_name] += 1
                            pass
                        pass
                    else:
                        pass

                if leg3in.intersects(circle):
                    if track.track_id in list_overlapping_leg1out:
                        l1_to_l3_count += 1
                        list_overlapping_leg1out.remove(track.track_id)
                        if class_name not in class_counter_1enter3:
                            class_counter_1enter3[class_name] = 1
                        else:
                            class_counter_1enter3[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg2out:
                        l2_to_l3_count += 1
                        list_overlapping_leg2out.remove(track.track_id)
                        if class_name not in class_counter_2enter3:
                            class_counter_2enter3[class_name] = 1
                        else:
                            class_counter_2enter3[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg3out:
                        l3_to_l3_count += 1
                        list_overlapping_leg3out.remove(track.track_id)
                        if class_name not in class_counter_3enter3:
                            class_counter_3enter3[class_name] = 1
                        else:
                            class_counter_3enter3[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg4out:
                        l4_to_l3_count += 1
                        list_overlapping_leg4out.remove(track.track_id)
                        if class_name not in class_counter_4enter3:
                            class_counter_4enter3[class_name] = 1
                        else:
                            class_counter_4enter3[class_name] += 1
                            pass
                        pass
                    else:
                        pass
                if leg4in.intersects(circle):
                    if track.track_id in list_overlapping_leg1out:
                        l1_to_l4_count += 1
                        list_overlapping_leg1out.remove(track.track_id)
                        if class_name not in class_counter_1enter4:
                            class_counter_1enter4[class_name] = 1
                        else:
                            class_counter_1enter4[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg2out:
                        l2_to_l4_count += 1
                        list_overlapping_leg2out.remove(track.track_id)
                        if class_name not in class_counter_2enter4:
                            class_counter_2enter4[class_name] = 1
                        else:
                            class_counter_2enter4[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg3out:
                        l3_to_l4_count += 1
                        list_overlapping_leg3out.remove(track.track_id)
                        if class_name not in class_counter_3enter4:
                            class_counter_3enter4[class_name] = 1
                        else:
                            class_counter_3enter4[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg4out:
                        l4_to_l4_count += 1
                        list_overlapping_leg4out.remove(track.track_id)
                        if class_name not in class_counter_4enter4:
                            class_counter_4enter4[class_name] = 1
                        else:
                            class_counter_4enter4[class_name] += 1
                            pass
                        pass
                    else:
                        pass
                    pass
                else:
                    pass
            

                # Calculate the sum of the four lists (OUT)
                total_sum1 = (l1_to_l2_count + l1_to_l1_count + l1_to_l3_count + l1_to_l4_count)
                total_sum2 = (l2_to_l1_count + l2_to_l2_count + l2_to_l3_count + l2_to_l4_count)
                total_sum3 = (l3_to_l1_count + l3_to_l2_count + l3_to_l3_count + l3_to_l4_count)
                total_sum4 = (l4_to_l1_count + l4_to_l2_count + l4_to_l3_count + l4_to_l4_count)

                text_draw1 = 'TOTAL LEG 1 OUT: ' + str(total_sum1) + ' , 2 OUT: ' + str(total_sum2)+ ' , 3 OUT: ' + str(total_sum3)+ ' , 4 OUT: ' + str(total_sum4)
                text_draw2 = 'TOTAL LEG 1 IN: ' + str((l1_to_l1_count+l2_to_l1_count+l3_to_l1_count+l4_to_l1_count)) + ' , 2 IN: ' + str((l1_to_l2_count+l2_to_l2_count+l3_to_l2_count+l4_to_l2_count))+ ' , 3 IN: ' + str((l1_to_l3_count+l2_to_l3_count+l3_to_l3_count+l4_to_l3_count))+ ' , 4 IN: ' + str((l1_to_l4_count+l2_to_l4_count+l3_to_l4_count+l4_to_l4_count))

                #Setting For total count text
                font_draw_number = cv2.FONT_HERSHEY_PLAIN
                draw_text_position1 = (int(width*0.01), int(height*0.98))
                draw_text_position2 = (int(width*0.01), int(height*0.05))
                cv2.putText(frame, text=text_draw1, org=draw_text_position1, fontFace=int(font_draw_number), fontScale=1, color=(255,255,255), thickness=2)
                cv2.putText(frame, text=text_draw2, org=draw_text_position2, fontFace=int(font_draw_number), fontScale=1, color=(255,255,255), thickness=2)

            #? If enable fivecounter flag
            if FLAGS.fivecounter:
                #Blue Line
                x1, y1 = 900,660
                x2, y2 = 800,680
                x3, y3 = 700,700
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 10)
                cv2.line(frame, (x2, y2), (x3, y3), (0, 0, 255), 10)
                cv2.putText(frame,('Leg 1'),(x1+20,y1+20), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,0,255),2)

                #Yellow Line
                x4, y4 = 180, 490
                x5, y5 = 185, 360
                x6, y6 = 190, 220
                cv2.line(frame, (x4, y4), (x5, y5), (255, 255, 0), 10)
                cv2.line(frame, (x5, y5), (x6, y6), (255, 255, 0), 10)
                cv2.putText(frame,('Leg 2'),(x4+20,y4+20), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(255,255,0),2)

                #Red Line
                x7, y7 = 380,50
                x8, y8 = 530,35
                x9, y9 = 680,20
                cv2.line(frame, (x7, y7), (x8, y8), (255, 0, 0), 10)
                cv2.line(frame, (x8, y8), (x9, y9), (255, 0, 0), 10)
                cv2.putText(frame,('Leg 3'),(x7+20,y7+20), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(255,0,0),2)

                #Green Line
                x10, y10 = 1040,160
                x11, y11 = 1080,300
                x12, y12 = 1130,420
                cv2.line(frame, (x10, y10), (x11, y11), (0, 255, 0), 10)
                cv2.line(frame, (x11, y11), (x12, y12), (0, 255, 0), 10)
                cv2.putText(frame,('Leg 4'),(x10+20,y10+20), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,255,0),2)

                #Purple Line
                x13, y13 = 22,700
                x14, y14 = 800,900
                x15, y15 = 1200,700
                cv2.line(frame, (x13, y13), (x14, y14), (255, 0, 255), 10)
                cv2.line(frame, (x14, y14), (x15, y15), (255, 0, 255), 10)
                cv2.putText(frame,('Leg 5'),(x13+20,y13+20), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(255,255,0),2)

                # Define the line segments
                leg1in = LineString([(x1, y1), (x2, y2)])
                leg1out = LineString([(x2, y2), (x3, y3)])
                leg2in = LineString([(x4, y4), (x5, y5)])
                leg2out = LineString([(x5, y5), (x6, y6)])
                leg3in = LineString([(x7, y7), (x8, y8)])
                leg3out = LineString([(x8, y8), (x9, y9)])
                leg4in = LineString([(x10, y10), (x11, y11)])
                leg4out = LineString([(x11, y11), (x12, y12)])
                leg5in = LineString([(x13, y13), (x14, y14)])
                leg5out = LineString([(x14, y14), (x15, y15)])

                # Define the circle
                circle = Point(x_center, y_center).buffer(radius)

                if leg1out.intersects(circle):
                    if track.track_id not in list_overlapping_leg1out:
                        list_overlapping_leg1out.append(track.track_id)
                    pass

                if leg2out.intersects(circle):
                    if track.track_id not in list_overlapping_leg2out:
                        list_overlapping_leg2out.append(track.track_id)
                    pass

                if leg3out.intersects(circle):
                    if track.track_id not in list_overlapping_leg3out:
                        list_overlapping_leg3out.append(track.track_id)
                    pass

                if leg4out.intersects(circle):
                    if track.track_id not in list_overlapping_leg4out:
                        list_overlapping_leg4out.append(track.track_id)
                    pass

                if leg5out.intersects(circle):
                    if track.track_id not in list_overlapping_leg5out:
                        list_overlapping_leg5out.append(track.track_id)
                    pass

                if leg1in.intersects(circle):
                    if track.track_id in list_overlapping_leg1out:
                        l1_to_l1_count += 1
                        list_overlapping_leg1out.remove(track.track_id)
                        if class_name not in class_counter_1enter1:
                            class_counter_1enter1[class_name] = 1
                        else:
                            class_counter_1enter1[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg2out:
                        l2_to_l1_count += 1
                        list_overlapping_leg2out.remove(track.track_id)
                        if class_name not in class_counter_2enter1:
                            class_counter_2enter1[class_name] = 1
                        else:
                            class_counter_2enter1[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg3out:
                        l3_to_l1_count += 1
                        list_overlapping_leg3out.remove(track.track_id)
                        if class_name not in class_counter_3enter1:
                            class_counter_3enter1[class_name] = 1
                        else:
                            class_counter_3enter1[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg4out:
                        l4_to_l1_count += 1
                        list_overlapping_leg4out.remove(track.track_id)
                        if class_name not in class_counter_4enter1:
                            class_counter_4enter1[class_name] = 1
                        else:
                            class_counter_4enter1[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg5out:
                        l5_to_l1_count += 1
                        list_overlapping_leg5out.remove(track.track_id)
                        if class_name not in class_counter_5enter1:
                            class_counter_5enter1[class_name] = 1
                        else:
                            class_counter_5enter1[class_name] += 1
                            pass
                        pass

                    else:
                        pass

                if leg2in.intersects(circle):
                    if track.track_id in list_overlapping_leg1out:
                        l1_to_l2_count += 1
                        list_overlapping_leg1out.remove(track.track_id)
                        if class_name not in class_counter_1enter2:
                            class_counter_1enter2[class_name] = 1
                        else:
                            class_counter_1enter2[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg2out:
                        l2_to_l2_count += 1
                        list_overlapping_leg2out.remove(track.track_id)
                        if class_name not in class_counter_2enter2:
                            class_counter_2enter2[class_name] = 1
                        else:
                            class_counter_2enter2[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg3out:
                        l3_to_l2_count += 1
                        list_overlapping_leg3out.remove(track.track_id)
                        if class_name not in class_counter_3enter2:
                            class_counter_3enter2[class_name] = 1
                        else:
                            class_counter_3enter2[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg4out:
                        l4_to_l2_count += 1
                        list_overlapping_leg4out.remove(track.track_id)
                        if class_name not in class_counter_4enter2:
                            class_counter_4enter2[class_name] = 1
                        else:
                            class_counter_4enter2[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg5out:
                        l5_to_l2_count += 1
                        list_overlapping_leg5out.remove(track.track_id)
                        if class_name not in class_counter_5enter2:
                            class_counter_5enter2[class_name] = 1
                        else:
                            class_counter_5enter2[class_name] += 1
                            pass
                        pass

                    else:
                        pass

                if leg3in.intersects(circle):
                    if track.track_id in list_overlapping_leg1out:
                        l1_to_l3_count += 1
                        list_overlapping_leg1out.remove(track.track_id)
                        if class_name not in class_counter_1enter3:
                            class_counter_1enter3[class_name] = 1
                        else:
                            class_counter_1enter3[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg2out:
                        l2_to_l3_count += 1
                        list_overlapping_leg2out.remove(track.track_id)
                        if class_name not in class_counter_2enter3:
                            class_counter_2enter3[class_name] = 1
                        else:
                            class_counter_2enter3[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg3out:
                        l3_to_l3_count += 1
                        list_overlapping_leg3out.remove(track.track_id)
                        if class_name not in class_counter_3enter3:
                            class_counter_3enter3[class_name] = 1
                        else:
                            class_counter_3enter3[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg4out:
                        l4_to_l3_count += 1
                        list_overlapping_leg4out.remove(track.track_id)
                        if class_name not in class_counter_4enter3:
                            class_counter_4enter3[class_name] = 1
                        else:
                            class_counter_4enter3[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg5out:
                        l5_to_l3_count += 1
                        list_overlapping_leg5out.remove(track.track_id)
                        if class_name not in class_counter_5enter3:
                            class_counter_5enter3[class_name] = 1
                        else:
                            class_counter_5enter3[class_name] += 1
                            pass
                        pass

                    else:
                        pass
                if leg4in.intersects(circle):
                    if track.track_id in list_overlapping_leg1out:
                        l1_to_l4_count += 1
                        list_overlapping_leg1out.remove(track.track_id)
                        if class_name not in class_counter_1enter4:
                            class_counter_1enter4[class_name] = 1
                        else:
                            class_counter_1enter4[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg2out:
                        l2_to_l4_count += 1
                        list_overlapping_leg2out.remove(track.track_id)
                        if class_name not in class_counter_2enter4:
                            class_counter_2enter4[class_name] = 1
                        else:
                            class_counter_2enter4[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg3out:
                        l3_to_l4_count += 1
                        list_overlapping_leg3out.remove(track.track_id)
                        if class_name not in class_counter_3enter4:
                            class_counter_3enter4[class_name] = 1
                        else:
                            class_counter_3enter4[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg4out:
                        l4_to_l4_count += 1
                        list_overlapping_leg4out.remove(track.track_id)
                        if class_name not in class_counter_4enter4:
                            class_counter_4enter4[class_name] = 1
                        else:
                            class_counter_4enter4[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg5out:
                        l5_to_l4_count += 1
                        list_overlapping_leg5out.remove(track.track_id)
                        if class_name not in class_counter_5enter4:
                            class_counter_5enter4[class_name] = 1
                        else:
                            class_counter_5enter4[class_name] += 1
                            pass
                        pass

                    else:
                        pass

                if leg5in.intersects(circle):
                    if track.track_id in list_overlapping_leg1out:
                        l1_to_l5_count += 1
                        list_overlapping_leg1out.remove(track.track_id)
                        if class_name not in class_counter_1enter5:
                            class_counter_1enter5[class_name] = 1
                        else:
                            class_counter_1enter5[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg2out:
                        l2_to_l5_count += 1
                        list_overlapping_leg2out.remove(track.track_id)
                        if class_name not in class_counter_2enter5:
                            class_counter_2enter5[class_name] = 1
                        else:
                            class_counter_2enter5[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg3out:
                        l3_to_l5_count += 1
                        list_overlapping_leg3out.remove(track.track_id)
                        if class_name not in class_counter_3enter5:
                            class_counter_3enter5[class_name] = 1
                        else:
                            class_counter_3enter5[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg4out:
                        l4_to_l5_count += 1
                        list_overlapping_leg4out.remove(track.track_id)
                        if class_name not in class_counter_4enter5:
                            class_counter_4enter5[class_name] = 1
                        else:
                            class_counter_4enter5[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg5out:
                        l5_to_l5_count += 1
                        list_overlapping_leg5out.remove(track.track_id)
                        if class_name not in class_counter_5enter5:
                            class_counter_5enter5[class_name] = 1
                        else:
                            class_counter_5enter5[class_name] += 1
                            pass
                        pass

                    else:
                        pass

                    pass
                else:
                    pass
            
                # Calculate the sum of the four lists (OUT)
                total_sum1 = (l1_to_l2_count + l1_to_l1_count + l1_to_l3_count + l1_to_l4_count + l1_to_l5_count)
                total_sum2 = (l2_to_l1_count + l2_to_l2_count + l2_to_l3_count + l2_to_l4_count + l2_to_l5_count)
                total_sum3 = (l3_to_l1_count + l3_to_l2_count + l3_to_l3_count + l3_to_l4_count + l3_to_l5_count)
                total_sum4 = (l4_to_l1_count + l4_to_l2_count + l4_to_l3_count + l4_to_l4_count + l4_to_l5_count)
                total_sum5 = (l5_to_l1_count + l5_to_l2_count + l5_to_l3_count + l5_to_l4_count + l5_to_l5_count)

                text_draw1 = 'TOTAL LEG 1 OUT: ' + str(total_sum1) + ' , 2 OUT: ' + str(total_sum2)+ ' , 3 OUT: ' + str(total_sum3)+ ' , 4 OUT: ' + str(total_sum4)+ ' , 5 OUT: ' + str(total_sum5)
                text_draw2 = 'TOTAL LEG 1 IN: ' + str((l1_to_l1_count+l2_to_l1_count+l3_to_l1_count+l4_to_l1_count+l5_to_l1_count)) + ' , 2 IN: ' + str((l1_to_l2_count+l2_to_l2_count+l3_to_l2_count+l4_to_l2_count+l5_to_l2_count))+ ' , 3 IN: ' + str((l1_to_l3_count+l2_to_l3_count+l3_to_l3_count+l4_to_l3_count+l5_to_l3_count))+ ' , 4 IN: ' + str((l1_to_l4_count+l2_to_l4_count+l3_to_l4_count+l4_to_l4_count+l5_to_l4_count))+ ' , 5 IN: ' + str((l1_to_l5_count+l2_to_l5_count+l3_to_l5_count+l4_to_l5_count+l5_to_l5_count))

                #Setting For total count text
                font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
                draw_text_position1 = (int(width*0.05), int(height*0.98))
                draw_text_position2 = (int(width*0.10), int(height*0.05))
                cv2.putText(frame, text=text_draw1, org=draw_text_position1, fontFace=int(font_draw_number), fontScale=1, color=(255,255,255), thickness=2)
                cv2.putText(frame, text=text_draw2, org=draw_text_position2, fontFace=int(font_draw_number), fontScale=1, color=(255,255,255), thickness=2)

            #? If enable counter flag
            if FLAGS.sixcounter:
                #Blue Line
                x1, y1 = 860,670
                x2, y2 = 660,685
                x3, y3 = 450,700
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 10)
                cv2.line(frame, (x2, y2), (x3, y3), (0, 0, 255), 10)
                cv2.putText(frame,('Leg 1'),(x1+20,y1+20), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,0,255),2)
                            
                #Yellow Line
                x4, y4 = 160, 490
                x5, y5 = 165, 350
                x6, y6 = 170, 210
                cv2.line(frame, (x4, y4), (x5, y5), (255, 255, 0), 10)
                cv2.line(frame, (x5, y5), (x6, y6), (255, 255, 0), 10)
                cv2.putText(frame,('Leg 2'),(x4+20,y4+20), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(255,255,0),2)

                #Red Line
                x7, y7 = 270,125
                x8, y8 = 350,80
                x9, y9 = 430,40
                cv2.line(frame, (x7, y7), (x8, y8), (255, 0, 0), 10)
                cv2.line(frame, (x8, y8), (x9, y9), (255, 0, 0), 10)
                cv2.putText(frame,('Leg 3'),(x7+20,y7+20), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(255,0,0),2)

                #Green Line
                x10, y10 = 460,15
                x11, y11 = 620,18
                x12, y12 = 840,20
                cv2.line(frame, (x10, y10), (x11, y11), (0, 255, 0), 10)
                cv2.line(frame, (x11, y11), (x12, y12), (0, 255, 0), 10)
                cv2.putText(frame,('Leg 4'),(x10+20,y10+20), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,255,0),2)

                #Purple Line
                x13, y13 = 1020,100
                x14, y14 = 1130,205
                x15, y15 = 1160,305
                cv2.line(frame, (x13, y13), (x14, y14), (255, 0, 255), 10)
                cv2.line(frame, (x14, y14), (x15, y15), (255, 0, 255), 10)
                cv2.putText(frame,('Leg 5'),(x13+20,y13+20), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(255, 0, 255),2)

                #Brown Line
                x16, y16 = 1160,400
                x17, y17 = 1080,520
                x18, y18 = 995,640
                cv2.line(frame, (x16, y16), (x17, y17), (153, 76, 0), 10)
                cv2.line(frame, (x17, y17), (x18, y18), (153, 76, 0), 10)
                cv2.putText(frame,('Leg 6'),(x16+20,y16+20), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(153, 76, 0), 2)

                # Define the line segments
                leg1in = LineString([(x1, y1), (x2, y2)])
                leg1out = LineString([(x2, y2), (x3, y3)])
                leg2in = LineString([(x4, y4), (x5, y5)])
                leg2out = LineString([(x5, y5), (x6, y6)])
                leg3in = LineString([(x7, y7), (x8, y8)])
                leg3out = LineString([(x8, y8), (x9, y9)])
                leg4in = LineString([(x10, y10), (x11, y11)])
                leg4out = LineString([(x11, y11), (x12, y12)])
                leg5in = LineString([(x13, y13), (x14, y14)])
                leg5out = LineString([(x14, y14), (x15, y15)])
                leg6in = LineString([(x16, y16), (x17, y17)])
                leg6out = LineString([(x17, y17), (x18, y18)])


                # Define the circle
                circle = Point(x_center, y_center).buffer(radius)

                if leg1out.intersects(circle):
                    if track.track_id not in list_overlapping_leg1out:
                        list_overlapping_leg1out.append(track.track_id)
                    pass

                if leg2out.intersects(circle):
                    if track.track_id not in list_overlapping_leg2out:
                        list_overlapping_leg2out.append(track.track_id)
                    pass

                if leg3out.intersects(circle):
                    if track.track_id not in list_overlapping_leg3out:
                        list_overlapping_leg3out.append(track.track_id)
                    pass

                if leg4out.intersects(circle):
                    if track.track_id not in list_overlapping_leg4out:
                        list_overlapping_leg4out.append(track.track_id)
                    pass

                if leg5out.intersects(circle):
                    if track.track_id not in list_overlapping_leg5out:
                        list_overlapping_leg5out.append(track.track_id)
                    pass

                if leg6out.intersects(circle):
                    if track.track_id not in list_overlapping_leg6out:
                        list_overlapping_leg6out.append(track.track_id)
                    pass

                if leg1in.intersects(circle):
                    if track.track_id in list_overlapping_leg1out:
                        l1_to_l1_count += 1
                        list_overlapping_leg1out.remove(track.track_id)
                        if class_name not in class_counter_1enter1:
                            class_counter_1enter1[class_name] = 1
                        else:
                            class_counter_1enter1[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg2out:
                        l2_to_l1_count += 1
                        list_overlapping_leg2out.remove(track.track_id)
                        if class_name not in class_counter_2enter1:
                            class_counter_2enter1[class_name] = 1
                        else:
                            class_counter_2enter1[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg3out:
                        l3_to_l1_count += 1
                        list_overlapping_leg3out.remove(track.track_id)
                        if class_name not in class_counter_3enter1:
                            class_counter_3enter1[class_name] = 1
                        else:
                            class_counter_3enter1[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg4out:
                        l4_to_l1_count += 1
                        list_overlapping_leg4out.remove(track.track_id)
                        if class_name not in class_counter_4enter1:
                            class_counter_4enter1[class_name] = 1
                        else:
                            class_counter_4enter1[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg5out:
                        l5_to_l1_count += 1
                        list_overlapping_leg5out.remove(track.track_id)
                        if class_name not in class_counter_5enter1:
                            class_counter_5enter1[class_name] = 1
                        else:
                            class_counter_5enter1[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg6out:
                        l6_to_l1_count += 1
                        list_overlapping_leg6out.remove(track.track_id)
                        if class_name not in class_counter_6enter1:
                            class_counter_6enter1[class_name] = 1
                        else:
                            class_counter_6enter1[class_name] += 1
                            pass
                        pass

                    else:
                        pass

                if leg2in.intersects(circle):
                    if track.track_id in list_overlapping_leg1out:
                        l1_to_l2_count += 1
                        list_overlapping_leg1out.remove(track.track_id)
                        if class_name not in class_counter_1enter2:
                            class_counter_1enter2[class_name] = 1
                        else:
                            class_counter_1enter2[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg2out:
                        l2_to_l2_count += 1
                        list_overlapping_leg2out.remove(track.track_id)
                        if class_name not in class_counter_2enter2:
                            class_counter_2enter2[class_name] = 1
                        else:
                            class_counter_2enter2[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg3out:
                        l3_to_l2_count += 1
                        list_overlapping_leg3out.remove(track.track_id)
                        if class_name not in class_counter_3enter2:
                            class_counter_3enter2[class_name] = 1
                        else:
                            class_counter_3enter2[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg4out:
                        l4_to_l2_count += 1
                        list_overlapping_leg4out.remove(track.track_id)
                        if class_name not in class_counter_4enter2:
                            class_counter_4enter2[class_name] = 1
                        else:
                            class_counter_4enter2[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg5out:
                        l5_to_l2_count += 1
                        list_overlapping_leg5out.remove(track.track_id)
                        if class_name not in class_counter_5enter2:
                            class_counter_5enter2[class_name] = 1
                        else:
                            class_counter_5enter2[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg6out:
                        l6_to_l2_count += 1
                        list_overlapping_leg6out.remove(track.track_id)
                        if class_name not in class_counter_6enter2:
                            class_counter_6enter2[class_name] = 1
                        else:
                            class_counter_6enter2[class_name] += 1
                            pass
                        pass

                    else:
                        pass

                if leg3in.intersects(circle):
                    if track.track_id in list_overlapping_leg1out:
                        l1_to_l3_count += 1
                        list_overlapping_leg1out.remove(track.track_id)
                        if class_name not in class_counter_1enter3:
                            class_counter_1enter3[class_name] = 1
                        else:
                            class_counter_1enter3[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg2out:
                        l2_to_l3_count += 1
                        list_overlapping_leg2out.remove(track.track_id)
                        if class_name not in class_counter_2enter3:
                            class_counter_2enter3[class_name] = 1
                        else:
                            class_counter_2enter3[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg3out:
                        l3_to_l3_count += 1
                        list_overlapping_leg3out.remove(track.track_id)
                        if class_name not in class_counter_3enter3:
                            class_counter_3enter3[class_name] = 1
                        else:
                            class_counter_3enter3[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg4out:
                        l4_to_l3_count += 1
                        list_overlapping_leg4out.remove(track.track_id)
                        if class_name not in class_counter_4enter3:
                            class_counter_4enter3[class_name] = 1
                        else:
                            class_counter_4enter3[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg5out:
                        l5_to_l3_count += 1
                        list_overlapping_leg5out.remove(track.track_id)
                        if class_name not in class_counter_5enter3:
                            class_counter_5enter3[class_name] = 1
                        else:
                            class_counter_5enter3[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg6out:
                        l6_to_l3_count += 1
                        list_overlapping_leg6out.remove(track.track_id)
                        if class_name not in class_counter_6enter3:
                            class_counter_6enter3[class_name] = 1
                        else:
                            class_counter_6enter3[class_name] += 1
                            pass
                        pass

                    else:
                        pass
                if leg4in.intersects(circle):
                    if track.track_id in list_overlapping_leg1out:
                        l1_to_l4_count += 1
                        list_overlapping_leg1out.remove(track.track_id)
                        if class_name not in class_counter_1enter4:
                            class_counter_1enter4[class_name] = 1
                        else:
                            class_counter_1enter4[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg2out:
                        l2_to_l4_count += 1
                        list_overlapping_leg2out.remove(track.track_id)
                        if class_name not in class_counter_2enter4:
                            class_counter_2enter4[class_name] = 1
                        else:
                            class_counter_2enter4[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg3out:
                        l3_to_l4_count += 1
                        list_overlapping_leg3out.remove(track.track_id)
                        if class_name not in class_counter_3enter4:
                            class_counter_3enter4[class_name] = 1
                        else:
                            class_counter_3enter4[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg4out:
                        l4_to_l4_count += 1
                        list_overlapping_leg4out.remove(track.track_id)
                        if class_name not in class_counter_4enter4:
                            class_counter_4enter4[class_name] = 1
                        else:
                            class_counter_4enter4[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg5out:
                        l5_to_l4_count += 1
                        list_overlapping_leg5out.remove(track.track_id)
                        if class_name not in class_counter_5enter4:
                            class_counter_5enter4[class_name] = 1
                        else:
                            class_counter_5enter4[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg6out:
                        l6_to_l4_count += 1
                        list_overlapping_leg6out.remove(track.track_id)
                        if class_name not in class_counter_6enter4:
                            class_counter_6enter4[class_name] = 1
                        else:
                            class_counter_6enter4[class_name] += 1
                            pass
                        pass

                    else:
                        pass

                if leg5in.intersects(circle):
                    if track.track_id in list_overlapping_leg1out:
                        l1_to_l5_count += 1
                        list_overlapping_leg1out.remove(track.track_id)
                        if class_name not in class_counter_1enter5:
                            class_counter_1enter5[class_name] = 1
                        else:
                            class_counter_1enter5[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg2out:
                        l2_to_l5_count += 1
                        list_overlapping_leg2out.remove(track.track_id)
                        if class_name not in class_counter_2enter5:
                            class_counter_2enter5[class_name] = 1
                        else:
                            class_counter_2enter5[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg3out:
                        l3_to_l5_count += 1
                        list_overlapping_leg3out.remove(track.track_id)
                        if class_name not in class_counter_3enter5:
                            class_counter_3enter5[class_name] = 1
                        else:
                            class_counter_3enter5[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg4out:
                        l4_to_l5_count += 1
                        list_overlapping_leg4out.remove(track.track_id)
                        if class_name not in class_counter_4enter5:
                            class_counter_4enter5[class_name] = 1
                        else:
                            class_counter_4enter5[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg5out:
                        l5_to_l5_count += 1
                        list_overlapping_leg5out.remove(track.track_id)
                        if class_name not in class_counter_5enter5:
                            class_counter_5enter5[class_name] = 1
                        else:
                            class_counter_5enter5[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg6out:
                        l6_to_l5_count += 1
                        list_overlapping_leg6out.remove(track.track_id)
                        if class_name not in class_counter_6enter5:
                            class_counter_6enter5[class_name] = 1
                        else:
                            class_counter_6enter5[class_name] += 1
                            pass
                        pass

                    else:
                        pass

                if leg6in.intersects(circle):
                    if track.track_id in list_overlapping_leg1out:
                        l1_to_l6_count += 1
                        list_overlapping_leg1out.remove(track.track_id)
                        if class_name not in class_counter_1enter6:
                            class_counter_1enter6[class_name] = 1
                        else:
                            class_counter_1enter6[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg2out:
                        l2_to_l6_count += 1
                        list_overlapping_leg2out.remove(track.track_id)
                        if class_name not in class_counter_2enter6:
                            class_counter_2enter6[class_name] = 1
                        else:
                            class_counter_2enter6[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg3out:
                        l3_to_l6_count += 1
                        list_overlapping_leg3out.remove(track.track_id)
                        if class_name not in class_counter_3enter6:
                            class_counter_3enter6[class_name] = 1
                        else:
                            class_counter_3enter6[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg4out:
                        l4_to_l6_count += 1
                        list_overlapping_leg4out.remove(track.track_id)
                        if class_name not in class_counter_4enter6:
                            class_counter_4enter6[class_name] = 1
                        else:
                            class_counter_4enter6[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg5out:
                        l5_to_l6_count += 1
                        list_overlapping_leg5out.remove(track.track_id)
                        if class_name not in class_counter_5enter6:
                            class_counter_5enter6[class_name] = 1
                        else:
                            class_counter_5enter6[class_name] += 1
                            pass
                        pass

                    if track.track_id in list_overlapping_leg6out:
                        l6_to_l6_count += 1
                        list_overlapping_leg6out.remove(track.track_id)
                        if class_name not in class_counter_6enter6:
                            class_counter_6enter6[class_name] = 1
                        else:
                            class_counter_6enter6[class_name] += 1
                            pass
                        pass

                    else:
                        pass

                    pass
                else:
                    pass
           
                # Calculate the sum of the four lists (OUT)
                total_sum1 = (l1_to_l2_count + l1_to_l1_count + l1_to_l3_count + l1_to_l4_count + l1_to_l5_count + l1_to_l6_count)
                total_sum2 = (l2_to_l1_count + l2_to_l2_count + l2_to_l3_count + l2_to_l4_count + l2_to_l5_count + l2_to_l6_count)
                total_sum3 = (l3_to_l1_count + l3_to_l2_count + l3_to_l3_count + l3_to_l4_count + l3_to_l5_count + l3_to_l6_count)
                total_sum4 = (l4_to_l1_count + l4_to_l2_count + l4_to_l3_count + l4_to_l4_count + l4_to_l5_count + l4_to_l6_count)
                total_sum5 = (l5_to_l1_count + l5_to_l2_count + l5_to_l3_count + l5_to_l4_count + l5_to_l5_count + l5_to_l6_count)
                total_sum6 = (l6_to_l1_count + l6_to_l2_count + l6_to_l3_count + l6_to_l4_count + l6_to_l5_count + l6_to_l6_count)

                text_draw1 = 'TOTAL 1 OUT: ' + str(total_sum1) + ' , 2 OUT: ' + str(total_sum2)+ ' , 3 OUT: ' + str(total_sum3)+ ' , 4 OUT: ' + str(total_sum4)+ ' , 5 OUT: ' + str(total_sum5)+ ' , 6 OUT: ' + str(total_sum6)
                text_draw2 = 'TOTAL 1 IN: ' + str((l1_to_l1_count+l2_to_l1_count+l3_to_l1_count+l4_to_l1_count+l5_to_l1_count+l5_to_l1_count)) + ' , 2 IN: ' + str((l1_to_l2_count+l2_to_l2_count+l3_to_l2_count+l4_to_l2_count+l5_to_l2_count+l6_to_l2_count))+ ' , 3 IN: ' + str((l1_to_l3_count+l2_to_l3_count+l3_to_l3_count+l4_to_l3_count+l5_to_l3_count+l6_to_l3_count))+ ' , 4 IN: ' + str((l1_to_l4_count+l2_to_l4_count+l3_to_l4_count+l4_to_l4_count+l5_to_l4_count+l6_to_l4_count))+ ' , 5 IN: ' + str((l1_to_l5_count+l2_to_l5_count+l3_to_l5_count+l4_to_l5_count+l5_to_l5_count+l6_to_l5_count))+ ' , 6 IN: ' + str((l1_to_l6_count+l2_to_l6_count+l3_to_l6_count+l4_to_l6_count+l5_to_l6_count+l6_to_l6_count))

                #Setting For total count text
                font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
                draw_text_position1 = (int(width*0.02), int(height*0.98))
                draw_text_position2 = (int(width*0.02), int(height*0.05))
                cv2.putText(frame, text=text_draw1, org=draw_text_position1, fontFace=int(font_draw_number), fontScale=1, color=(255,255,255), thickness=2)
                cv2.putText(frame, text=text_draw2, org=draw_text_position2, fontFace=int(font_draw_number), fontScale=1, color=(255,255,255), thickness=2)

                #cv2.putText(frame, f'From Leg 4 to Leg 6', (5, 305), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
                #cv2.putText(frame, f'From Leg 6 to Leg 2', (width - 350, 305), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

                #idx = 0
                #for class_name, count in class_counter_4enter6.items():
                    #cnt_str = f'{class_name}: {count}'
                    #cv2.line(frame, (20, 65 + (idx*40)), (240, 65 + (idx*40)), [0,255,255], 25)
                    #cv2.putText(frame, cnt_str, (20, 75 + (idx*40)), 0, 1, [0,0,255], thickness = 1, lineType = cv2.LINE_AA)
                    #idx += 1
 
                #idx = 0
                #for class_name, count in class_counter_6enter2.items():
                    #cnt_str2 = f'{class_name}: {count}'
                    #cv2.line(frame, (width - 240, 65 + (idx*40)), (width-20, 65 + (idx*40)), [0,255,255], 25)
                    #cv2.putText(frame, cnt_str2, (width - 240, 75 + (idx*40)), 0, 1, [0,0,255], thickness = 1, lineType = cv2.LINE_AA)
                    #idx += 1

                #idx = 0
                #for class_name, count in class_counter_1enter3.items():
                    #cnt_str3 = f'{class_name}: {count}'
                    #cv2.line(frame, (20, 500 + (idx*40)), (240, 500 + (idx*40)), [0,255,255], 25)
                    #cv2.putText(frame, cnt_str3, (20, 510 + (idx*40)), 0, 1, [0,0,255], thickness = 1, lineType = cv2.LINE_AA)
                    #idx += 1

                #idx = 0
                #for class_name, count in class_counter_1enter4.items():
                    #cnt_str4 = f'{class_name}: {count}'
                    #cv2.line(frame, (width - 240, 500 + (idx*40)), (width-20, 500 + (idx*40)), [0,255,255], 25)
                    #cv2.putText(frame, cnt_str4, (width - 240, 510 + (idx*40)), 0, 1, [0,0,255], thickness = 1, lineType = cv2.LINE_AA)
                    #idx += 1


            # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))


        for i in range(frame_num):
            if FLAGS.fourcounter:
                print('Frame #: ', frame_num)
                # calculate frames per second of running detections
                fps = 1.0 / (time.time() - start_time)
                print("FPS: %.2f" % fps)
                result = np.asarray(frame)
                result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                print ("--------------------------------")
                print("1 Enter 1:", class_counter_1enter1)
                print("1 Enter 2:", class_counter_1enter2)
                print("1 Enter 3:", class_counter_1enter3)
                print("1 Enter 4:", class_counter_1enter4)
                print("2 Enter 1:", class_counter_2enter1)
                print("2 Enter 2:", class_counter_2enter2)
                print("2 Enter 3:", class_counter_2enter3)
                print("2 Enter 4:", class_counter_2enter4)
                print("3 Enter 1:", class_counter_3enter1)
                print("3 Enter 2:", class_counter_3enter2)
                print("3 Enter 3:", class_counter_3enter3)
                print("3 Enter 4:", class_counter_3enter4)
                print("4 Enter 1:", class_counter_4enter1)
                print("4 Enter 2:", class_counter_4enter2)
                print("4 Enter 3:", class_counter_4enter3)
                print("4 Enter 4:", class_counter_4enter4)
                print ("--------------------------------")

                last_frame_summary = f"Summary for frame {frame_num}\n1 Enter 1: {class_counter_1enter1}\n1 Enter 2: {class_counter_1enter2}\n1 Enter 3: {class_counter_1enter3}\n1 Enter 4: {class_counter_1enter4}\n\n2 Enter 1: {class_counter_2enter1}\n2 Enter 2: {class_counter_2enter2}\n2 Enter 3: {class_counter_2enter3}\n2 Enter 4: {class_counter_2enter4}\n\n3 Enter 1: {class_counter_3enter1}\n3 Enter 2: {class_counter_3enter2}\n3 Enter 3: {class_counter_3enter3}\n3 Enter 4: {class_counter_3enter4}\n\n4 Enter 1: {class_counter_4enter1}\n4 Enter 2: {class_counter_4enter2}\n4 Enter 3: {class_counter_4enter3}\n4 Enter 4: {class_counter_4enter4}\n"

                # print last frame summary again
                print("Last frame summary:")
                print(last_frame_summary)

                # Create a dictionary with the summary data
                summary_data = {
                    '1 Enter 1': class_counter_1enter1,
                    '1 Enter 2': class_counter_1enter2,
                    '1 Enter 3': class_counter_1enter3,
                    '1 Enter 4': class_counter_1enter4,
                    '2 Enter 1': class_counter_2enter1,
                    '2 Enter 2': class_counter_2enter2,
                    '2 Enter 3': class_counter_2enter3,
                    '2 Enter 4': class_counter_2enter4,
                    '3 Enter 1': class_counter_3enter1,
                    '3 Enter 2': class_counter_3enter2,
                    '3 Enter 3': class_counter_3enter3,
                    '3 Enter 4': class_counter_3enter4,
                    '4 Enter 1': class_counter_4enter1,
                    '4 Enter 2': class_counter_4enter2,
                    '4 Enter 3': class_counter_4enter3,
                    '4 Enter 4': class_counter_4enter4,
                    }
                df = pd.DataFrame(summary_data)
                df = df.transpose()
                df.to_excel('result.xlsx', header=allowed_classes, index_label='Turning Movement')

            if FLAGS.fivecounter:
                print('Frame #: ', frame_num)
                # calculate frames per second of running detections
                fps = 1.0 / (time.time() - start_time)
                print("FPS: %.2f" % fps)
                result = np.asarray(frame)
                result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                print ("--------------------------------")
                print("1 Enter 1:", class_counter_1enter1)
                print("1 Enter 2:", class_counter_1enter2)
                print("1 Enter 3:", class_counter_1enter3)
                print("1 Enter 4:", class_counter_1enter4)
                print("1 Enter 5:", class_counter_1enter5)
                print("2 Enter 1:", class_counter_2enter1)
                print("2 Enter 2:", class_counter_2enter2)
                print("2 Enter 3:", class_counter_2enter3)
                print("2 Enter 4:", class_counter_2enter4)
                print("2 Enter 5:", class_counter_2enter5)
                print("3 Enter 1:", class_counter_3enter1)
                print("3 Enter 2:", class_counter_3enter2)
                print("3 Enter 3:", class_counter_3enter3)
                print("3 Enter 4:", class_counter_3enter4)
                print("3 Enter 5:", class_counter_3enter5)
                print("4 Enter 1:", class_counter_4enter1)
                print("4 Enter 2:", class_counter_4enter2)
                print("4 Enter 3:", class_counter_4enter3)
                print("4 Enter 4:", class_counter_4enter4)
                print("4 Enter 5:", class_counter_4enter5)
                print("5 Enter 1:", class_counter_5enter1)
                print("5 Enter 2:", class_counter_5enter2)
                print("5 Enter 3:", class_counter_5enter3)
                print("5 Enter 4:", class_counter_5enter4)
                print("5 Enter 5:", class_counter_5enter5)
                print ("--------------------------------")

                last_frame_summary = f"Summary for frame {frame_num}\n1 Enter 1: {class_counter_1enter1}\n1 Enter 2: {class_counter_1enter2}\n1 Enter 3: {class_counter_1enter3}\n1 Enter 4: {class_counter_1enter4}\n1 Enter 5: {class_counter_1enter5}\n\n2 Enter 1: {class_counter_2enter1}\n2 Enter 2: {class_counter_2enter2}\n2 Enter 3: {class_counter_2enter3}\n2 Enter 4: {class_counter_2enter4}\n2 Enter 5: {class_counter_2enter5}\n\n3 Enter 1: {class_counter_3enter1}\n3 Enter 2: {class_counter_3enter2}\n3 Enter 3: {class_counter_3enter3}\n3 Enter 4: {class_counter_3enter4}\n3 Enter 5: {class_counter_3enter5}\n\n\n4 Enter 1: {class_counter_4enter1}\n4 Enter 2: {class_counter_4enter2}\n4 Enter 3: {class_counter_4enter3}\n4 Enter 4: {class_counter_4enter4}\n4 Enter 5: {class_counter_4enter5}\n\n5 Enter 1: {class_counter_5enter1}\n5 Enter 2: {class_counter_5enter2}\n5 Enter 3: {class_counter_5enter3}\n5 Enter 4: {class_counter_5enter4}\n5 Enter 5: {class_counter_5enter5}\n"

                # print last frame summary again
                print("Last frame summary:")
                print(last_frame_summary)

                # Create a dictionary with the summary data
                summary_data = {
                    '1 Enter 1': class_counter_1enter1,
                    '1 Enter 2': class_counter_1enter2,
                    '1 Enter 3': class_counter_1enter3,
                    '1 Enter 4': class_counter_1enter4,
                    '1 Enter 5': class_counter_1enter5,
                    '2 Enter 1': class_counter_2enter1,
                    '2 Enter 2': class_counter_2enter2,
                    '2 Enter 3': class_counter_2enter3,
                    '2 Enter 4': class_counter_2enter4,
                    '2 Enter 5': class_counter_2enter5,
                    '3 Enter 1': class_counter_3enter1,
                    '3 Enter 2': class_counter_3enter2,
                    '3 Enter 3': class_counter_3enter3,
                    '3 Enter 4': class_counter_3enter4,
                    '3 Enter 5': class_counter_3enter5,
                    '4 Enter 1': class_counter_4enter1,
                    '4 Enter 2': class_counter_4enter2,
                    '4 Enter 3': class_counter_4enter3,
                    '4 Enter 4': class_counter_4enter4,
                    '4 Enter 5': class_counter_4enter5,
                    '5 Enter 1': class_counter_5enter1,
                    '5 Enter 2': class_counter_5enter2,
                    '5 Enter 3': class_counter_5enter3,
                    '5 Enter 4': class_counter_5enter4,
                    '5 Enter 5': class_counter_5enter5,
                    }
                df = pd.DataFrame(summary_data)
                df = df.transpose()
                df.to_excel('result.xlsx', header=allowed_classes, index_label='Turning Movement')
            
            if FLAGS.sixcounter:
                print('Frame #: ', frame_num)
                # calculate frames per second of running detections
                fps = 1.0 / (time.time() - start_time)
                print("FPS: %.2f" % fps)
                result = np.asarray(frame)
                result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                print ("--------------------------------")
                print("1 Enter 1:", class_counter_1enter1)
                print("1 Enter 2:", class_counter_1enter2)
                print("1 Enter 3:", class_counter_1enter3)
                print("1 Enter 4:", class_counter_1enter4)
                print("1 Enter 5:", class_counter_1enter5)
                print("1 Enter 6:", class_counter_1enter6)
                print("2 Enter 1:", class_counter_2enter1)
                print("2 Enter 2:", class_counter_2enter2)
                print("2 Enter 3:", class_counter_2enter3)
                print("2 Enter 4:", class_counter_2enter4)
                print("2 Enter 5:", class_counter_2enter5)
                print("2 Enter 6:", class_counter_2enter6)
                print("3 Enter 1:", class_counter_3enter1)
                print("3 Enter 2:", class_counter_3enter2)
                print("3 Enter 3:", class_counter_3enter3)
                print("3 Enter 4:", class_counter_3enter4)
                print("3 Enter 5:", class_counter_3enter5)
                print("3 Enter 6:", class_counter_3enter6)
                print("4 Enter 1:", class_counter_4enter1)
                print("4 Enter 2:", class_counter_4enter2)
                print("4 Enter 3:", class_counter_4enter3)
                print("4 Enter 4:", class_counter_4enter4)
                print("4 Enter 5:", class_counter_4enter5)
                print("4 Enter 6:", class_counter_4enter6)
                print("5 Enter 1:", class_counter_5enter1)
                print("5 Enter 2:", class_counter_5enter2)
                print("5 Enter 3:", class_counter_5enter3)
                print("5 Enter 4:", class_counter_5enter4)
                print("5 Enter 5:", class_counter_5enter5)
                print("5 Enter 6:", class_counter_5enter6)
                print("6 Enter 1:", class_counter_6enter1)
                print("6 Enter 2:", class_counter_6enter2)
                print("6 Enter 3:", class_counter_6enter3)
                print("6 Enter 4:", class_counter_6enter4)
                print("6 Enter 5:", class_counter_6enter5)
                print("6 Enter 6:", class_counter_6enter6)
                print ("--------------------------------")

                last_frame_summary = f"Summary for frame {frame_num}\n1 Enter 1: {class_counter_1enter1}\n1 Enter 2: {class_counter_1enter2}\n1 Enter 3: {class_counter_1enter3}\n1 Enter 4: {class_counter_1enter4}\n1 Enter 5: {class_counter_1enter5}\n1 Enter 6: {class_counter_1enter6}\n\n2 Enter 1: {class_counter_2enter1}\n2 Enter 2: {class_counter_2enter2}\n2 Enter 3: {class_counter_2enter3}\n2 Enter 4: {class_counter_2enter4}\n2 Enter 5: {class_counter_2enter5}\n2 Enter 6: {class_counter_2enter6}\n\n3 Enter 1: {class_counter_3enter1}\n3 Enter 2: {class_counter_3enter2}\n3 Enter 3: {class_counter_3enter3}\n3 Enter 4: {class_counter_3enter4}\n3 Enter 5: {class_counter_3enter5}\n3 Enter 6: {class_counter_3enter6}\n\n4 Enter 1: {class_counter_4enter1}\n4 Enter 2: {class_counter_4enter2}\n4 Enter 3: {class_counter_4enter3}\n4 Enter 4: {class_counter_4enter4}\n4 Enter 5: {class_counter_4enter5}\n4 Enter 6: {class_counter_4enter6}\n\n5 Enter 1: {class_counter_5enter1}\n5 Enter 2: {class_counter_5enter2}\n5 Enter 3: {class_counter_5enter3}\n5 Enter 4: {class_counter_5enter4}\n5 Enter 5: {class_counter_5enter5}\n5 Enter 6: {class_counter_5enter6}\n\n6 Enter 1: {class_counter_6enter1}\n6 Enter 2: {class_counter_6enter2}\n6 Enter 3: {class_counter_6enter3}\n6 Enter 4: {class_counter_6enter4}\n6 Enter 5: {class_counter_6enter5}\n6 Enter 6: {class_counter_6enter6}\n"

                # print last frame summary again
                print("Last frame summary:")
                print(last_frame_summary)

                # Create a dictionary with the summary data
                summary_data = {
                    '1 Enter 1': class_counter_1enter1,
                    '1 Enter 2': class_counter_1enter2,
                    '1 Enter 3': class_counter_1enter3,
                    '1 Enter 4': class_counter_1enter4,
                    '1 Enter 5': class_counter_1enter5,
                    '1 Enter 6': class_counter_1enter6,
                    '2 Enter 1': class_counter_2enter1,
                    '2 Enter 2': class_counter_2enter2,
                    '2 Enter 3': class_counter_2enter3,
                    '2 Enter 4': class_counter_2enter4,
                    '2 Enter 5': class_counter_2enter5,
                    '2 Enter 6': class_counter_2enter6,
                    '3 Enter 1': class_counter_3enter1,
                    '3 Enter 2': class_counter_3enter2,
                    '3 Enter 3': class_counter_3enter3,
                    '3 Enter 4': class_counter_3enter4,
                    '3 Enter 5': class_counter_3enter5,
                    '3 Enter 6': class_counter_3enter6,
                    '4 Enter 1': class_counter_4enter1,
                    '4 Enter 2': class_counter_4enter2,
                    '4 Enter 3': class_counter_4enter3,
                    '4 Enter 4': class_counter_4enter4,
                    '4 Enter 5': class_counter_4enter5,
                    '4 Enter 6': class_counter_4enter6,
                    '5 Enter 1': class_counter_5enter1,
                    '5 Enter 2': class_counter_5enter2,
                    '5 Enter 3': class_counter_5enter3,
                    '5 Enter 4': class_counter_5enter4,
                    '5 Enter 5': class_counter_5enter5,
                    '5 Enter 6': class_counter_5enter6,
                    '6 Enter 1': class_counter_6enter1,
                    '6 Enter 2': class_counter_6enter2,
                    '6 Enter 3': class_counter_6enter3,
                    '6 Enter 4': class_counter_6enter4,
                    '6 Enter 5': class_counter_6enter5,
                    '6 Enter 6': class_counter_6enter6,
                    }

                df = pd.DataFrame(summary_data)
                df = df.transpose()

            # Make sure the length of allowed_classes matches the number of columns in the DataFrame
            #if len(allowed_classes) != len(df.columns):
                # Modify the allowed_classes list to match the number of columns
                #allowed_classes = df.columns.tolist()
                df.to_excel('result.xlsx', header=allowed_classes, index_label='Turning Movement')



        #if not FLAGS.fourcounter and not FLAGS.fivecounter and not FLAGS.sixcounter:
            #result = np.asarray(frame)
            #result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not FLAGS.dont_show:
            #print("Width:", width)
            #print("Height:", height)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Display', width, height)
            cv2.imshow('Display', frame)
            #resized_frame = cv2.resize(frame, (width, height))
            #cv2.imshow('Zoomed Out Video', resized_frame)
            #cv2.imshow("Output Video", result)
                    # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

    pass
pass


cv2.destroyAllWindows()
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
       
