########################################
# Author : alivcor (Abhinandan Dubey)
# Stony Brook university

import cv2, numpy as np
from vgg16_feat import VGG16
from keras.models import Model
from keras.layers import Dense
import glob
import progressbar

def _preprocess_image(input_image):
    top_left = input_image[0:224, 0:224].astype(np.float32)
    top_right = input_image[288:288+224,0:224].astype(np.float32)
    bottom_right = input_image[288:288+224, 288:288+224].astype(np.float32)
    bottom_left = input_image[0:224, 288:288+224].astype(np.float32)
    center = input_image[146:146+224, 146:146+224].astype(np.float32)
    flipped_input_image = cv2.flip(input_image, 0)
    flipped_top_left = flipped_input_image[0:224, 0:224].astype(np.float32)
    flipped_top_right = flipped_input_image[288:288 + 224, 0:224].astype(np.float32)
    flipped_bottom_right = flipped_input_image[288:288 + 224, 288:288 + 224].astype(np.float32)
    flipped_bottom_left = flipped_input_image[0:224, 288:288 + 224].astype(np.float32)
    flipped_center = flipped_input_image[146:146 + 224, 146:146 + 224].astype(np.float32)
    # print top_left.shape
    # print top_right.shape
    # print bottom_right.shape
    # print bottom_left.shape
    # print center.shape
    input_tensor = (top_left + top_right + bottom_right + bottom_left + center + flipped_top_left + flipped_top_right + flipped_bottom_right + flipped_bottom_left + flipped_center)/10
    # input_tensor = np.mean(np.array([ old_set, new_set ]), axis=0 )
    processed_input_tensor = np.expand_dims(input_tensor.astype(np.float32), axis=0)
    # print "Preprocessing Complete !"
    # print "processed_input_tensor.shape : ", processed_input_tensor.shape
    return processed_input_tensor


def init_load_subopt_model():
    vgg_model = VGG16(include_top=True, weights='imagenet')
    x = Dense(9, activation='softmax', name='predictions')(vgg_model.layers[-2].output)
    # Then create the corresponding model
    subopt_vgg = Model(input=[vgg_model.input], output=x)
    subopt_vgg.summary()
    return subopt_vgg


def fine_tune_network(network, trainX, trainY):
    network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    network.fit(trainX, trainY, batch_size=10, epochs=700)
    return network


def process_experiment(experiment_id):
    image_list = glob.glob("dataset/" + str(experiment_id) + "/*.png")
    processed_images = []
    processed_outputs = []
    image_count = 0
    with progressbar.ProgressBar(max_value=len(image_list)) as progress:
        for image_file_path in image_list:
            image = cv2.imread(image_file_path, flags=cv2.IMREAD_COLOR)
            processed_image = _preprocess_image(image)
            processed_images.append(processed_image)
            onehot_output = np.zeros(9)
            onehot_output[experiment_id - 1] = 1.
            processed_outputs.append(onehot_output)
            image_count += 1
            progress.update(image_count)
    processed_images = np.squeeze(np.array(processed_images))
    processed_outputs = np.squeeze(np.array(processed_outputs))
    print processed_images.shape, processed_outputs.shape
    return processed_images, processed_outputs


def preprocess_data(num_experiments):
    experiment_ids = range(1,num_experiments+1)
    trainX = []
    trainY = []
    for experiment_id in experiment_ids:
        print "Processing experiment ", experiment_id
        processed_images, processed_outputs = process_experiment(experiment_id)
        trainX.append(processed_images)
        trainY.append(processed_outputs)

    trainX = np.reshape(np.squeeze(np.array(trainX)), (-1, 224, 224, 3))
    trainY = np.reshape(np.squeeze(np.array(trainY)),  (-1, 9))

    print "Processed all experiments."
    print trainX.shape, trainY.shape
    return trainX, trainY


subopt_model = init_load_subopt_model()
trainX, trainY = preprocess_data(9)
tuned_network = fine_tune_network(subopt_model, trainX, trainY)