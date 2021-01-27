  # import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
# import pathlib
# import tensorflow as tf

# tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# # Enable GPU dynamic memory allocation
# # gpus = tf.config.experimental.list_physical_devices('GPU')
# # for gpu in gpus:
# #     tf.config.experimental.set_memory_growth(gpu, True)

# PATH_TO_SAVED_MODEL = 'F:/Cynapto/DOCKER/object_detection_model_pre_trained/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8/saved_model'
# # PATH_TO_LABELS = 'F:/Cynapto/DOCKER/object_detection_model_pre_trained/mscoco_label_map.pbtxt'



# # def download_labels(filename):
# #     base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
# #     label_dir = tf.keras.utils.get_file(fname=filename,
# #                                         origin=base_url + filename,
# #                                         untar=False)
# #     label_dir = pathlib.Path(label_dir)
# #     return str(label_dir)	

# # LABEL_FILENAME = 'mscoco_label_map.pbtxt'
# PATH_TO_LABELS = "F:/Cynapto/DOCKER/object_detection_model_pre_trained/mscoco_label_map.pbtxt"
# # PATH_TO_LABELS = download_labels(LABEL_FILENAME)
  


def load_image_into_numpy_array(path):

    return np.array(Image.open(path))







def image(img_folder):


  IMAGE_PATHS = []
  # img_folder = 'F:/Cynapto/DOCKER/object_detection_model_pre_trained/image/'
  for i in os.listdir(img_folder):
    IMAGE_PATHS.append(img_folder+i)


  for image_path in IMAGE_PATHS:

      print('Running inference for {}... '.format(image_path), end='')

      image_np = load_image_into_numpy_array(image_path)

      input_tensor = tf.convert_to_tensor(image_np)
      # The model expects a batch of images, so add an axis with `tf.newaxis`.
      input_tensor = input_tensor[tf.newaxis, ...]

      # input_tensor = np.expand_dims(image_np, 0)
      detections = detect_fn(input_tensor)
      num_detections = int(detections.pop('num_detections'))
      detections = {key: value[0, :num_detections].numpy()
                     for key, value in detections.items()}
      detections['num_detections'] = num_detections

      # detection_classes should be ints.
      detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

      image_np_with_detections = image_np.copy()

      viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.60,
            agnostic_mode=False)

      # plt.figure()
      # plt.imshow(image_np_with_detections)
      cv2.imwrite(image_path+'detected.jpg',image_np_with_detections)
      # plt.imsave(image_path+'detect.jpg',image_np_with_detections)
      cv2.imshow('detected',image_np_with_detections)
      print('Done')
  # plt.show()


  # pass

def video(vid_file):
  out_path = "/opt/detected_video.mp4"
  fourcc = cv2.VideoWriter_fourcc(*'avc1') 
  

  cap = cv2.VideoCapture(vid_file)

  # out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width,  height))

  writer = None
  while cap.isOpened():
    ret, image_np = cap.read()

    # print('Running inference for {}... '.format(image_path), end='')

    # image_np = load_image_into_numpy_array(image_path)

    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.50,
          agnostic_mode=False)


    if writer is None:
        # create VideoWrite object
      writer = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS),(640,640))
    writer.write(image_np_with_detections)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if cv2.waitKey(1) == ord('q'):
          break
    cv2.imshow('object_detection', cv2.resize(image_np_with_detections,(600,600)))
    if cv2.waitKey(25) & 0xff == ord('q'):
      cv2.destroyAllWindows()
      break

  cap.release()
  cv2.destroyAllWindows()
  writer.release() 

def webcam(): 
  out_path = "/opt/detected_video.mp4"
  fourcc = cv2.VideoWriter_fourcc(*'avc1') 
  

  cap = cv2.VideoCapture(0)

  # out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width,  height))

  writer = None
  while cap.isOpened():
    ret, image_np = cap.read()

    # print('Running inference for {}... '.format(image_path), end='')

    # image_np = load_image_into_numpy_array(image_path)

    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.50,
          agnostic_mode=False)


    if writer is None:
        # create VideoWrite object
      writer = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS),(640,640))
    writer.write(image_np_with_detections)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if cv2.waitKey(1) == ord('q'):
          break
    cv2.imshow('object_detection', cv2.resize(image_np_with_detections,(600,600)))
    if cv2.waitKey(25) & 0xff == ord('q'):
      cv2.destroyAllWindows()
      break

  cap.release()
  cv2.destroyAllWindows()
  writer.release() 




def default():
  print('inside default function')



import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf

#tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# PATH_TO_SAVED_MODEL = 'F:/Cynapto/DOCKER/object_detection_model_pre_trained/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8/saved_model'
# PATH_TO_LABELS = 'F:/Cynapto/DOCKER/object_detection_model_pre_trained/mscoco_label_map.pbtxt'


import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils



# PATH_TO_LABELS = "F:/Cynapto/DOCKER/object_detection_model_pre_trained/mscoco_label_map.pbtxt"



# category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
#                                                                     use_display_name=True)
# print('done')
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings


if __name__ == "__main__":


  #PATH_TO_LABELS = "F:/Cynapto/DOCKER/object_detection_ubuntu/mscoco_label_map.pbtxt"

  PATH_TO_LABELS = "/opt/mscoco_label_map.pbtxt"

  #PATH_TO_SAVED_MODEL = 'F:/Cynapto/DOCKER/object_detection_ubuntu/ssd_mobilenet/saved_model/'

  PATH_TO_SAVED_MODEL = "/opt/ssd_mobilenet/saved_model"


  img_folder = "/opt/image/"

  #img_folder = 'F:/Cynapto/DOCKER/object_detection_ubuntu/image/'

  # PATH_TO_SAVED_MODEL = os.environ['model']
  # PATH_TO_LABELS = os.environ['label']
  # image_path = os.environ['img_file']
  choice = int(os.environ['choice'])
  img_folder = os.environ['img_folder']
  vid_file=os.environ['vid_file']


  print('Loading model...', end='')
  start_time = time.time()

  detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

  end_time = time.time()
  elapsed_time = end_time - start_time
  print('Done! Took {} seconds'.format(elapsed_time))


  category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                      use_display_name=True)
  print('choice =',type(choice))
  if choice==0:
    # pass
    image(img_folder)
  elif choice==1:
    video(vid_file)
  elif choice==2:
    webcam()
  else:
    default()
