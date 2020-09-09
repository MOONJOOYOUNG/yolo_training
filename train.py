from absl import app, flags, logging
from absl.flags import FLAGS
import os
import shutil
import tensorflow as tf
from core.yolov4 import YOLO, decode, compute_loss, decode_train
from core.dataset_tiny import Dataset
from core.config import cfg
import numpy as np
from core import utils
from core.utils import freeze_all, unfreeze_all
import pdb

flags.DEFINE_string('model', 'yolov4', 'yolov4, yolov3')
flags.DEFINE_string('weights', './scripts/yolov4-tiny.weights', 'pretrained weights')
# flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')

#app._run_init(['yolov4'], app.parse_flags_with_usage)

def main(_argv):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # 텐서플로가 첫 번째 GPU만 사용하도록 제한
        try:
            tf.config.experimental.set_visible_devices(gpus[7], 'GPU')
        except RuntimeError as e:
            # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
            print(e)
    
    trainset = Dataset(FLAGS, is_training=True)
    #print(next(iter(trainset)))
    #_train = next(iter(trainset)) #_train[0] = (3, 416, 416, 3), len(train[1] =3,

    #len(train[1][0])) =2,

    # _train[1][0][0].shape : (3, 26, 26, 3, 25)
    # _train[1][0][1].shape : (3, 150, 4)

    # len(_train[1][1]) =2,
    # _train[1][1][0].shape : (3, 26, 26, 3, 25)
    # _train[1][1][1].shape : (3, 150, 4)

    testset = Dataset(FLAGS, is_training=False)
    logdir = "./data/log"
    isfreeze = False
    steps_per_epoch = len(trainset)
    first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
    second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
    total_steps = (first_stage_epochs + second_stage_epochs) * steps_per_epoch
    # train_steps = (first_stage_epochs + second_stage_epochs) * steps_per_period

    input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH

    freeze_layers = utils.load_freeze_layer(FLAGS.model, FLAGS.tiny)
    #yolov4 : ['conv2d_93', 'conv2d_101', 'conv2d_109']
    #yolov4_tiny: ['conv2d_17', 'conv2d_20']

    feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)
# '''
#     import core.backbone as backbone
#     route_1, route_2, conv = backbone.cspdarknet53(input_layer)

#     route_1
#     Out[97]: <tf.Tensor 'Mul_355:0' shape=(None, 52, 52, 256) dtype=float32>

#     route_2
#     Out[98]: <tf.Tensor 'Mul_376:0' shape=(None, 26, 26, 512) dtype=float32>

#     conv
#     Out[99]: <tf.Tensor 'LeakyRelu_164:0' shape=(None, 13, 13, 512) dtype=float32>

# '''
    #yolov4 : [<tf.Tensor 'conv2d_93/BiasAdd:0' shape=(None, 52, 52, 75) dtype=float32>,
    #<tf.Tensor 'conv2d_101/BiasAdd:0' shape=(None, 26, 26, 75) dtype=float32>,
    #<tf.Tensor 'conv2d_109/BiasAdd:0' shape=(None, 13, 13, 75) dtype=float32>]

    #yolov4_tiny : [<tf.Tensor 'conv2d_130/BiasAdd:0' shape=(None, 26, 26, 75) dtype=float32>,
    #<tf.Tensor 'conv2d_127/BiasAdd:0' shape=(None, 13, 13, 75) dtype=float32>]

    if FLAGS.tiny:
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            else:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)
    else:
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            elif i == 1:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            else:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    model.summary()
    # from contextlib import redirect_stdout
    # with open('modelsummary.txt', 'w') as f:
    #     with redirect_stdout(f):
    #         model.summary()

    if FLAGS.weights == None:
        print("Training from scratch")
    else:
        if FLAGS.weights.split(".")[len(FLAGS.weights.split(".")) - 1] == "weights":
            utils.load_weights(model, FLAGS.weights, FLAGS.model, FLAGS.tiny)
        else:
            model.load_weights(FLAGS.weights)
        print('Restoring weights from: %s ... ' % FLAGS.weights)

    # wf = open(FLAGS.weights, 'rb')
    # major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
    # wf.close()
    # layer_size = 110
    # output_pos = [93, 101, 109]
    # j = 0
    # for i in range(layer_size):
    #     conv_layer_name = 'conv2d_%d' %i if i > 0 else 'conv2d'
    #     bn_layer_name = 'batch_normalization_%d' %j if j > 0 else 'batch_normalization'
    #     conv_layer = model.get_layer(conv_layer_name)
    #     filters = conv_layer.filters
    #     k_size = conv_layer.kernel_size[0]
    #     in_dim = conv_layer.input_shape[-1]



    optimizer = tf.keras.optimizers.Adam()
    if os.path.exists(logdir): shutil.rmtree(logdir)
    writer = tf.summary.create_file_writer(logdir)

    # define training step function
    # @tf.function
    def train_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(len(freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                # print(pred)

                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            tf.print("=> STEP %4d/%4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, total_steps, optimizer.lr.numpy(),
                                                               giou_loss, conf_loss,
                                                               prob_loss, total_loss))
            # update learning rate
            global_steps.assign_add(1)
            if global_steps < warmup_steps:
                lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
            else:
                lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
                )
            optimizer.lr.assign(lr.numpy())

            # writing summary data
            with writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
            writer.flush()
    def test_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(len(freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            tf.print("=> TEST STEP %4d   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, giou_loss, conf_loss,
                                                               prob_loss, total_loss))

    for epoch in range(first_stage_epochs + second_stage_epochs):
        if epoch < first_stage_epochs:
            if not isfreeze:
                isfreeze = True
                for name in freeze_layers:
                    freeze = model.get_layer(name)
                    freeze_all(freeze)
        elif epoch >= first_stage_epochs:
            if isfreeze:
                isfreeze = False
                for name in freeze_layers:
                    freeze = model.get_layer(name)
                    unfreeze_all(freeze)
        for image_data, target in trainset:
            train_step(image_data, target)
        #for image_data, target in testset:
        #    test_step(image_data, target)
        model.save_weights("./checkpoints/yolov4")

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
