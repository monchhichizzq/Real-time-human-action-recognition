import os
import cv2
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
from sklearn import metrics
import matplotlib.pyplot as plt
import keras.backend as K
from keras.callbacks import LearningRateScheduler

def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  OK  ---")

    else:
        print("---  There is this folder!  ---")

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 200:
        lr *= 1e-2
    elif epoch > 100:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def monitoring(save_dir, batch_size, model_name):
    # Prepare model model saving directory.
    # save_dir = os.path.join('../model_save', 'cifar_5')
    model_name_save = 'HG-'+model_name+'-{val_accuracy:.2f}_' + '.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name_save)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=10,
                                   min_lr=0.5e-6)

    #board = TensorBoard(log_dir='../model_save/logs')
    log_dir_name = "logs/{}".format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    tensorboard = TensorBoard(log_dir=log_dir_name,
                              histogram_freq=100,
                              batch_size=batch_size,
                              write_graph=False,
                              write_grads=False,
                              write_images=False,
                              embeddings_freq=0,
                              embeddings_layer_names=None,
                              embeddings_metadata=None,
                              embeddings_data=None,
                              update_freq='epoch')

    callbacks = [checkpoint, tensorboard]
    #callbacks = [tensorboard]
    return callbacks

def output_nb_neurons(model):
    nb_neurons = 0
    for layer in model.layers:
        if layer.name.startswith('conv2d'):
            outputs = layer.output
            # The amount of neurons after each conv layer equals fp-height*fp-width*fp-channels
            nb_neurons_per_layer = outputs.shape[1] * outputs.shape[2] *outputs.shape[3]
            nb_neurons = nb_neurons + nb_neurons_per_layer
            print( '=====>>>[ Number of neurons (',layer.name,'):', nb_neurons_per_layer, ' ]', )
        elif layer.name.startswith('activation'):
            outputs = layer.output
            # The amount of neurons after each conv layer equals fp-height*fp-width*fp-channels
            if  len(outputs.shape)== 4:
                nb_neurons_per_layer = outputs.shape[1] * outputs.shape[2]*outputs.shape[3]
                nb_neurons = nb_neurons + nb_neurons_per_layer
                print(  '=====>>>[ Number of neurons (',layer.name,'):', nb_neurons_per_layer, ' ]')
            else:
                nb_neurons_per_layer = outputs.shape[1]
                nb_neurons = nb_neurons + nb_neurons_per_layer
                print( '=====>>>[ Number of neurons (',layer.name,'):',nb_neurons_per_layer, ' ]')
        elif layer.name.startswith('average'):
            outputs = layer.output
            # The amount of neurons after each conv layer equals fp-height*fp-width*fp-channels
            nb_neurons_per_layer = outputs.shape[1] * outputs.shape[2]*outputs.shape[3]
            nb_neurons = nb_neurons + nb_neurons_per_layer
            print( '=====>>>[ Number of neurons (',layer.name,'):', nb_neurons_per_layer, ' ]')
        elif layer.name.startswith('global'):
            outputs = layer.output
            # The amount of neurons after each conv layer equals fp-height*fp-width*fp-channels
            nb_neurons_per_layer = outputs.shape[1]
            nb_neurons = nb_neurons + nb_neurons_per_layer
            print('=====>>>[ Number of neurons (global_average):', nb_neurons_per_layer, ' ]')
        elif layer.name.startswith('dense'):
            outputs = layer.output
            nb_neurons_per_layer = outputs.shape[1]
            nb_neurons = nb_neurons + nb_neurons_per_layer
            print( '=====>>>[ Number of neurons (',layer.name,'):', nb_neurons_per_layer, ' ]')
        elif layer.name.startswith('main_input'):
            outputs = layer.output
            # The amount of neurons after each conv layer equals fp-height*fp-width*fp-channels
            nb_neurons_per_layer = outputs.shape[1] * outputs.shape[2]
            nb_neurons = nb_neurons + nb_neurons_per_layer
            print( '=====>>>[ Number of neurons (',layer.name,'):', nb_neurons_per_layer, ' ]')
    return nb_neurons

def tf_gpus_options():
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
    #print(gpus, cpus)
    ### By default, TensorFlow requests and blocks nearly all of the GPU memory of the two available GPUs to avoid memory fragmentation
    ### Ask for memory only when you needed
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    ### Limit the consumption of memory use to 2GB
    # tf.config.experimental.set_virtual_device_configuration(
    #     gpus[0],
    #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
    # )
    ### Create two virtual gpu in one GPU
    # tf.config.experimental.set_virtual_device_configuration(
    #     gpus[0],
    #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048),
    #      tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

def plot_confusion_matrix(model_name, y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel = 'True label',
           xlabel = 'Predicted label\naccuracy={:0.2f}; misclass={:0.2f}'.format(accuracy, misclass))

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    #plt.ylabel('True label', fontsize=8)
    #plt.xlabel('Predicted label\naccuracy={:0.2f}; misclass={:0.2f}'.format(accuracy, misclass), fontsize=8)
    #plt.colorbar()
    fig = plt.gcf()
    fig.savefig(model_name + '.png', dpi=600)
    return ax, cm

def zoom_out_dataset(x, y):
    x_processed = []
    for i in range(len(x)):
        image = x[i]
        image_processed = zoom_out(image, label=y[i].argmax())
        x_processed.append(image_processed)
    return x_processed, y

def zoom_out(image, label):
    new_image = cv2.resize(image,(32,24),interpolation=cv2.INTER_LANCZOS4)
    image = np.zeros((48, 64))
    if label == 0:
        image[12:36, 16:48] = new_image
    elif label == 1:
        image[12:36, 32:64] = new_image
    elif label == 2:
        image[12:36, 0:32] = new_image
    return image

def thr_list_shift(thr_plot, num_thrs, num_epoch):
    thrs = []
    for j in range(num_thrs):
        thrs_per_epoch = []
        for i in range(num_epoch):
            thrs_j = thr_plot[i][j]
            thrs_per_epoch.append(thrs_j)
        thrs.append(thrs_per_epoch)
    return thrs

def new_thr(thr):
    log2_thr = np.ceil(np.log2(thr))
    return 2**(log2_thr)

def import_quantized_thrs(epoch, bit = 11):
    y_thr = np.ones((len(epoch), ))
    thr_2bit = [2**i for i in range(bit)]
    y_thr_2bit = [y_thr*factor for factor in thr_2bit[4:]]
    return y_thr_2bit

def acc_monitoring(epoch, state_bit, state_bit_standard, post_train_acc, post_acc, acc, acc_C, acc_L, acc_R, post_sat_loss_plot, post_train_sat_loss_plot, total_loss, cat_loss, train_acc, train_loss_plot,  train_sat_loss_plot, test_sat_loss_plot, save_path, thr_plot, state_bits_plot, layer_names):
    ### Train: Green(#15b01a), aqua(#13eac9), sea green(#53fca1), aquamarine, grass green(#3f9b0b), forest green(#154406), mint green(#9ffeb0), spring green(#a9f971)
    ### Test: Blue, light blue(#95d0fc), sky blue(#448ee4), royal blue(#0504aa), bright blue(#0165fc),  pale blue(#d0fefe), electric blue(#0652ff), darker blue(#00035b)
    ### Sat_loss: Yellow, bright yellow(#fffd01), golden yellow(#fac205), lemon(#fdff52), shit
    ### Deep layer:
    ### Sallow layer:

    plt.figure(11)
    num_epoch = len(epoch)
    num_state = np.array(state_bits_plot).shape[1]
    y_state = np.ones((len(epoch), ))*state_bit
    y_state_standard = np.ones((len(epoch), ))*state_bit_standard
    state_bits_plot = thr_list_shift(state_bits_plot, num_state, num_epoch)
    for i in range(num_state):
        plt.plot(epoch,  state_bits_plot[i], label=layer_names[i], linewidth=0.5)
    plt.plot(epoch, y_state, label="$ state \quad bit $", linewidth=0.5, linestyle= '--')
    plt.plot(epoch,  y_state_standard, label="$ state \quad bit \quad standard$", linewidth=0.5, linestyle='--')
    plt.legend(loc='upper left', fontsize=8)
    plt.xlabel('Epoch')
    plt.ylabel('Max state bits - log2')
    fig = plt.gcf()
    #plt.pause(1)
    fig.savefig(save_path + '/Max_state_bits.png', dpi=600)
    plt.close()

    plt.figure(11)
    num_epoch = len(epoch)
    num_thrs = np.array(thr_plot).shape[1]
    thr_plot = thr_list_shift(thr_plot, num_thrs, num_epoch)
    y_thr_2bit = import_quantized_thrs(epoch, bit = 11)

    num_lines = np.array(y_thr_2bit).shape[0]
    # if thr_plot[0][0] > 512:
    for i in range(6):
        plt.plot(epoch, y_thr_2bit[i], color = 'gray', linewidth=0.5, linestyle='--')
        # plt.text(0, y_thr_2bit[i],  str(y_thr_2bit[i]))
    for i in range(num_thrs):
        plt.plot(epoch, thr_plot[i], label="thr"+str(i), linewidth=0.5)
    plt.legend(loc='upper left', fontsize=8)
    plt.xlabel('Epoch')
    plt.ylabel('Thr')
    fig = plt.gcf()
    #plt.pause(1)
    fig.savefig(save_path + '/Thresholds.png', dpi=600)
    plt.close()

    plt.figure(21)
    plt.subplot(211)
    plt.plot(epoch, train_acc, label="$Train \quad Accuracy$", color="green", linewidth=0.5)
    plt.plot(epoch, acc, label="$Test \quad Accuracy$", color="blue", linewidth=0.5)

    plt.plot(epoch, post_train_acc, label="$Train \quad Accuracy \quad after \quad quantization$", color="#448ee4", linewidth=0.5, linestyle= '--')
    plt.plot(epoch, post_acc, label="$Test \quad  Accuracy \quad after \quad quantization$", color="#53fca1", linewidth=0.5, linestyle='--')
    plt.legend(loc='upper left', fontsize=8)
    plt.ylim((60, 100))
    # plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(212)
    plt.plot(epoch, acc_C, label="$Clapping$", color="#448ee4", linewidth=0.5)
    plt.plot(epoch, acc_L, label="$Left \quad waving$", color="orange", linewidth=0.5)
    plt.plot(epoch, acc_R, label="$Right \quad waving$", color="yellowgreen", linewidth=0.5)
    plt.legend(loc='upper left', fontsize=8)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    fig = plt.gcf()
    #plt.pause(1)
    fig.savefig(save_path + '/plot_tiny.png', dpi=600)
    plt.close()

    plt.figure(21)
    plt.subplot(211)
    plt.plot(epoch, cat_loss, label="$Cat \quad loss$", color="#fac205", linewidth=0.5)
    plt.plot(epoch, total_loss, label="$Total \quad loss$", color="green", linewidth=0.5)
    plt.plot(epoch, train_loss_plot, label="$Train \quad loss$", color="blue", linewidth=0.5)
    plt.legend(loc='lower left', fontsize=8)
    # plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(212)
    plt.plot(epoch, train_sat_loss_plot, label="$Train \quad sat \quad loss$", color="#53fca1", linewidth=0.5)
    plt.plot(epoch, test_sat_loss_plot, label="$Test \quad sat \quad loss$", color="#0652ff", linewidth=0.5)
    # plt.plot(epoch, post_train_sat_loss_plot, label="$Post Train \quad sat \quad loss$", color="#9ffeb0", linewidth=1)
    # plt.plot(epoch, post_sat_loss_plot, label="$Post Test \quad sat \quad loss$", color="#d0fefe", linewidth=1)
    plt.legend(loc='upper left', fontsize=8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    fig = plt.gcf()
    #plt.pause(1)
    fig.savefig(save_path + '/plot_tiny_loss.png', dpi=600)
    plt.close()

    #plt.show()