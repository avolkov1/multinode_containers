from __future__ import print_function

import argparse
import keras
from keras import backend as K
from keras.preprocessing import image
from keras.datasets import fashion_mnist
# from keras_contrib.applications.wide_resnet import WideResidualNetwork
from wide_resnet_pad import WideResidualNetwork
# import numpy as np
import tensorflow as tf
# TODO: Step 1 work here: import horovod.keras
import horovod.keras as hvd
import os


def main():
    parser = argparse.ArgumentParser(
        description='Keras Fashion MNIST Example',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log-dir', default='./logs',
                        help='tensorboard log directory')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size for training')
    parser.add_argument('--val-batch-size', type=int, default=32,
                        help='input batch size for validation')
    parser.add_argument('--epochs', type=int, default=40,
                        help='number of epochs to train')
    parser.add_argument('--base-lr', type=float, default=0.01,
                        help='learning rate for a single GPU')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--wd', type=float, default=0.000005,
                        help='weight decay')
    # TODO: Step 9 part 1: register `--warmup-epochs`
    parser.add_argument('--warmup-epochs', type=float, default=5,
                        help='number of warmup epochs')

    GRAPHDEF_FILE = 'graphdef'
    parser.add_argument(
        '--savegraph', action='store', nargs='?',
        const=GRAPHDEF_FILE,
        help='Save graphdef pb and pbtxt files. '
        '(default: {})'.format(GRAPHDEF_FILE))

    parser.add_argument(
        '--profrun', action='store_true',
        help='Run for nsys/dlprof profiling. Runs only a few steps.')

    args = parser.parse_args()

    # Checkpoints will be written in the log directory.
    args.checkpoint_format = \
        os.path.join(args.log_dir, 'checkpoint-{epoch}.h5')

    print('AMP MIXED', os.environ.get("TF_ENABLE_AUTO_MIXED_PRECISION"))

    # TODO: Step 2 work here: initialize horovod
    hvd.init()

    # TODO: Step 3 work here: pin GPUs
    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    K.set_session(tf.Session(config=config))

    # If set > 0, will resume training from a given checkpoint.
    resume_from_epoch = 0
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
            resume_from_epoch = try_epoch
            break

    # TODO: Step 4 work here: broadcast `resume_from_epoch` from first process
    # to all others
    with tf.Session(config=config):
        resume_from_epoch = \
            hvd.broadcast(resume_from_epoch, 0, name='resume_from_epoch')

    # TODO: Step 5 work here: only set `verbose` to `1` if this is the
    # first worker
    verbose = 1 if hvd.rank() == 0 else 0

    # Input image dimensions
    img_rows, img_cols = 28, 28
    num_classes = 10

    # Download and load FASHION MNIST dataset.
    if hvd.rank() == 0:
        # Load Fashion MNIST data.
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    with tf.Session(config=config):
        # download/unzip in rank 0 only.
        hvd.allreduce([0], name="Barrier")

    if hvd.rank() != 0:
        # Load Fashion MNIST data.
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Training data iterator.
    train_gen = image.ImageDataGenerator(
        featurewise_center=True, featurewise_std_normalization=True,
        horizontal_flip=True, width_shift_range=0.2, height_shift_range=0.2)
    train_gen.fit(x_train)
    train_iter = train_gen.flow(x_train, y_train, batch_size=args.batch_size)

    # Validation data iterator.
    test_gen = image.ImageDataGenerator(
        featurewise_center=True, featurewise_std_normalization=True)
    test_gen.mean = train_gen.mean
    test_gen.std = train_gen.std
    test_iter = test_gen.flow(x_test, y_test, batch_size=args.val_batch_size)

    base_lr = args.base_lr
    LR = base_lr * hvd.size()

    # Restore from a previous checkpoint, if initial_epoch is specified.
    # if resume_from_epoch > 0 and hvd.rank() == 0:
    if resume_from_epoch > 0:
        # TODO: Step 6 work here: only execute the `if` statement if this is
        # the first worker
        # If this is only done in rank 0 get following errors:
        #     horovod/common/operations.cc:764] One or more tensors were
        #     submitted to be reduced, gathered or broadcasted by subset of
        #     ranks and are waiting for remainder of ranks
        model = keras.models.load_model(
            args.checkpoint_format.format(epoch=resume_from_epoch))
    else:
        # Set up standard WideResNet-16-10 model.
        model = WideResidualNetwork(
            depth=16, width=10, weights=None, input_shape=input_shape,
            classes=num_classes, dropout_rate=0.01)

        # WideResNet model that is included with Keras is optimized for
        # inference. Add L2 weight decay & adjust BN settings.
        model_config = model.get_config()
        for layer, layer_config in zip(model.layers, model_config['layers']):
            if hasattr(layer, 'kernel_regularizer'):
                regularizer = keras.regularizers.l2(args.wd)
                layer_config['config']['kernel_regularizer'] = \
                    {'class_name': regularizer.__class__.__name__,
                     'config': regularizer.get_config()}
            if type(layer) == keras.layers.BatchNormalization:
                layer_config['config']['momentum'] = 0.9
                layer_config['config']['epsilon'] = 1e-5

        model = keras.models.Model.from_config(model_config)

        # TODO: Step 7 part 1 work here: increase the base learning rate by the
        # number of workers
        opt = keras.optimizers.SGD(
            lr=LR, momentum=args.momentum)

        # TODO: Step 7 part 2 work here: Wrap the optimizer in a Horovod
        # distributed optimizer
        opt_dist = hvd.DistributedOptimizer(opt)

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=opt_dist,
                      metrics=['accuracy'])

    def lr_schedule(epoch):
        # global LR
        if epoch < 15:
            return LR
        if epoch < 25:
            return 1e-1 * LR
        if epoch < 35:
            return 1e-2 * LR
        return 1e-3 * LR

    warmup_epochs = args.warmup_epochs
    callbacks = [
        # TODO: Step 8: broadcast initial variable states from the first
        # worker to all others
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),

        # TODO: Step 12: average the metrics among workers at the end of every
        # epoch
        hvd.callbacks.MetricAverageCallback(),

        # TODO: Step 9 part 2: implement a LR warmup over `args.warmup_epochs`
        hvd.callbacks.LearningRateWarmupCallback(
            warmup_epochs=warmup_epochs, verbose=verbose),

        # TODO: Step 9 part 3: replace with the Horovod learning rate
        # scheduler, taking care not to start until after warmup is complete
        hvd.callbacks.LearningRateScheduleCallback(
            lr_schedule, start_epoch=warmup_epochs)
    ]

    if hvd.rank() == 0:
        # TODO: Step 10: only append these 2 callbacks to `callbacks` if they
        # are to be executed by the first worker
        callbacks.append(
            keras.callbacks.ModelCheckpoint(args.checkpoint_format))
        callbacks.append(keras.callbacks.TensorBoard(args.log_dir))

    # Train the model.
    number_of_workers = hvd.size()
    steps_per_epoch = len(train_iter) // number_of_workers
    validation_steps = 3 * len(test_iter) // number_of_workers

    # Train the model.
    if args.profrun:
        steps_per_epoch = 4

    model.fit_generator(train_iter,
                        # TODO: Step 11 part 1: keep the total number of steps
                        # the same in spite of an increased number of workers
                        steps_per_epoch=steps_per_epoch,
                        callbacks=callbacks,
                        epochs=args.epochs,
                        verbose=verbose,
                        workers=number_of_workers,
                        initial_epoch=resume_from_epoch,
                        validation_data=test_iter,
                        # TODO: Step 11 part 2: Set this value to be
                        # 3 * num_test_iterations / number_of_workers
                        validation_steps=validation_steps)

    # Evaluate the model on the full data set.
    score = model.evaluate_generator(test_iter, len(test_iter),
                                     workers=number_of_workers)

    if verbose:
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    if hvd.rank() == 0 and args.savegraph:
        graphdef_file = args.savegraph

        session = K.get_session()
        graph_def = session.graph.as_graph_def()
        with open('{}.pb'.format(graphdef_file), 'wb') as f:
            f.write(graph_def.SerializeToString())
        with open('{}.pbtxt'.format(graphdef_file), 'w') as f:
            f.write(str(graph_def))


if __name__ == '__main__':
    main()
