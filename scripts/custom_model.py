# -*- coding: utf-8 -*-
import numpy as np
import sys
import tensorflow as tf
import cv2
import os
import json

from utils import label_map_util
from utils import visualization_utils as vis_util

cur_dir = os.path.dirname(os.path.abspath(__file__))

PATH_TO_FROZEN_GRAPH = os.path.join(cur_dir, 'model/pit_model.pb')
PATH_TO_LABELS = os.path.join(cur_dir,'model/pit_label.pbtxt')

detection_graph = tf.Graph()
category_index = None

def load_model(model_path, label_path):
    
    if model_path is None or label_path is None:
        global PATH_TO_FROZEN_GRAPH
        global PATH_TO_LABELS
        
        model_path = PATH_TO_FROZEN_GRAPH
        label_path = PATH_TO_LABELS
    
    global detection_graph
    global category_index
    
    detection_graph = tf.Graph()
    
    category_index = label_map_util.create_category_index_from_labelmap(label_path, use_display_name=True)
   
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
def load_labels(label_path):
    labels = label_map_util.create_category_index_from_labelmap(label_path, use_display_name=True)
    return labels

def get_objects(image, scr_tres = 0.5):  
    image_np = image.copy()
    objects = []

    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
    
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
       
        image_np_expanded = np.expand_dims(image_np, axis=0)

        (_boxes, _scores, _classes, _num) = sess.run(
              [detection_boxes, detection_scores, detection_classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})

        h = image_np.shape[0]
        w = image_np.shape[1]
        for _box, _score, _class in  zip(_boxes[0], _scores[0], _classes[0]):
            if _score > scr_tres:
                y0, x0, y1, x1 = int(_box[0]*h), int(_box[1]*w),\
                int(_box[2]*h), int(_box[3]*w)
                objects.append((x0, y0, x1, y1, _score))

    return objects

def draw_rect(image):
    image_np = image.copy()
    
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
           
            image_np_expanded = np.expand_dims(image_np, axis=0)
        
            (_boxes, _scores, _classes, _num) = sess.run(
                  [detection_boxes, detection_scores, detection_classes, num_detections],
                  feed_dict={image_tensor: image_np_expanded})

            h = image_np.shape[0]
            w = image_np.shape[1]
            print('-'*20)
            print('Got new image:')
            for _box, _score, _class in  zip(_boxes[0], _scores[0], _classes[0]):
                if _score > 0.5:
                   y0, x0, y1, x1 = int(_box[0]*h), int(_box[1]*w),\
                   int(_box[2]*h), int(_box[3]*w)
                   print('({:4}:{:4}) ({:4}:{:4}), Probability: {:.4}'.format(x0, y0, x1, y1, _score))
            print('-'*20)
            vis_util.visualize_boxes_and_labels_on_image_array(
                  image_np,
                  np.squeeze(_boxes),
                  np.squeeze(_classes).astype(np.int32),
                  np.squeeze(_scores),
                  category_index,
                  use_normalized_coordinates=True,
                  line_thickness=8)
    return image_np

load_model(None, None)


if __name__ == '__main__':
    print('Start main from custom_model.py')
    image_path = 'test_image.jpg'
    image = cv2.imread(image_path)
    
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        imgs = draw_rect(image)
        for i, img in enumerate(imgs):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite('test_out' + str(i) + '.jpg', img)

