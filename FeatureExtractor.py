import cv2, numpy as np
from vgg16_feat import VGG16
from keras.models import Model
import progressbar
import EventIssuer

def _preprocess_input(input_tensor):
    # print "Preprocessing Input Frame..."
    processed_input_tensor = cv2.resize(input_tensor, (224, 224)).astype(np.float32)
    processed_input_tensor = np.expand_dims(processed_input_tensor, axis=0)
    # print "Preprocessing Complete !"
    # print "processed_input_tensor.shape : ", processed_input_tensor.shape
    return processed_input_tensor


def init_load_extractor_model(logfilename):
    EventIssuer.issueMessage("Loading Extractor Model..", logfilename, True)
    vgg_model = VGG16(include_top=True, weights='imagenet')
    EventIssuer.issueMessage("Shredding Softmax Layer..", logfilename, True)
    vgg_feature_vector = vgg_model.layers[-2].output
    # print vgg_feature_vector.shape
    feature_extraction_model = Model(input=[vgg_model.input], output=[vgg_feature_vector])
    feature_extraction_model.summary()
    EventIssuer.issueSuccess("Extractor Model Created", logfilename)
    return feature_extraction_model


def vgg_sieve(model, input_tensors, logfilename):
    feature_vectors = []

    num_frames = len(input_tensors)
    EventIssuer.issueMessage("Extracting Features..", logfilename, True)
    frame_count = -1
    with progressbar.ProgressBar(max_value=num_frames) as progress:
        for input_tensor in input_tensors:
            frame_count += 1
            processed_input_tensor = _preprocess_input(input_tensor)
            feature_vector = model.predict(processed_input_tensor)
            feature_vectors.append(feature_vector)
            progress.update(frame_count)
    EventIssuer.issueSuccess("Features Extracted !", logfilename)
    return np.array(feature_vectors)


def extract_features(filename, feature_extraction_model, logfilename, stride=100, max_frames=150):
    cap = cv2.VideoCapture(filename)
    # # Read the first frame of the video
    video_frameSequence = []
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = -1

    EventIssuer.issueMessage("Loading file to memory..", logfilename, True)
    with progressbar.ProgressBar(max_value=totalFrames) as progress:
        while True:
            try:
                frame_count += 1
                if (frame_count % stride == 0 and frame_count <= max_frames):
                    ret, frame = cap.read()
                    # print frame.shape
                    video_frameSequence.append(frame)
                progress.update(frame_count)
            except AttributeError:
                break
            except ValueError:
                break

    EventIssuer.issueSuccess("Reading frames to memory complete.", logfilename)
    EventIssuer.issueMessage("Starting feature extraction.", logfilename, True)

    vgg_features = vgg_sieve(feature_extraction_model, video_frameSequence, logfilename)

    return vgg_features