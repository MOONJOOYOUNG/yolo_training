import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
from core.yolov4 import YOLO, decode, filter_boxes
import core.utils as utils
from core.config import cfg

#flags.DEFINE_string('weights', './data/yolov4.weights', 'path to weights file')
# flags.DEFINE_string('weights', './checkpoints/chck2/yolov4', 'path to weights file')
# flags.DEFINE_string('output', './checkpoints/yolov4-chck2', 'path to output')
# flags.DEFINE_boolean('tiny', False, 'is yolo-tiny or not')
# flags.DEFINE_integer('input_size', 416, 'define input size of export model')
# flags.DEFINE_float('score_thres', 0.2, 'define score threshold')
# flags.DEFINE_string('framework', 'tf', 'define what framework do you want to convert (tf, trt, tflite)')
# flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')

#yolov4_tiny
flags.DEFINE_string('weights', './checkpoints/yolov4_tiny_org1/yolov4', 'path to weights file')
flags.DEFINE_string('output', './checkpoints/yolov4_tiny_chck1', 'path to output')
flags.DEFINE_boolean('tiny', True, 'is yolo-tiny or not')
flags.DEFINE_integer('input_size', 416, 'define input size of export model')
flags.DEFINE_float('score_thres', 0.2, 'define score threshold')
flags.DEFINE_string('framework', 'tflite', 'define what framework do you want to convert (tf, trt, tflite)')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')

def save_tf():
  STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)

  input_layer = tf.keras.layers.Input([FLAGS.input_size, FLAGS.input_size, 3])
  feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)
  bbox_tensors = []
  prob_tensors = []
  if FLAGS.tiny:
    for i, fm in enumerate(feature_maps):
      if i == 0:
        output_tensors = decode(fm, FLAGS.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
      else:
        output_tensors = decode(fm, FLAGS.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
      bbox_tensors.append(output_tensors[0])
      prob_tensors.append(output_tensors[1])
  else:
    for i, fm in enumerate(feature_maps):
      if i == 0:
        output_tensors = decode(fm, FLAGS.input_size // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
      elif i == 1:
        output_tensors = decode(fm, FLAGS.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
      else:
        output_tensors = decode(fm, FLAGS.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
      bbox_tensors.append(output_tensors[0])
      prob_tensors.append(output_tensors[1])
  pred_bbox = tf.concat(bbox_tensors, axis=1)
  pred_prob = tf.concat(prob_tensors, axis=1)
  if FLAGS.framework == 'tflite':
    pred = (pred_bbox, pred_prob)
  else:
    boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=FLAGS.score_thres, input_shape=tf.constant([FLAGS.input_size, FLAGS.input_size]))
    pred = tf.concat([boxes, pred_conf], axis=-1)
  model = tf.keras.Model(input_layer, pred)
  model.load_weights(FLAGS.weights)
  #utils.load_weights(model, FLAGS.weights, FLAGS.model, FLAGS.tiny) #weight파일일 경
  
  #model.summary()
  #model.save('/checkpoints/yolov4-416')
  #model.save(FLAGS.output, save_format = 'tf')
  tf.saved_model.save(model, FLAGS.output) #현재 이 저장 형태의 경우: assets/, variables/, saved_model.pb 형태로 저장됨.
  #model.save('./checkpoints/yolov4-416.h5') #  이 방식의 경우 원하는 형식으로 저장않하지 않음.
  

def main(_argv):
  save_tf()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
