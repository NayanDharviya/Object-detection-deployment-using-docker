#FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
FROM ubuntu
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update -y

RUN apt install -y python3-pip python3-dev
RUN python3 -m pip install --upgrade pip



#COPY requirement.txt /opt/requirements.txt
#WORKDIR /opt/
#RUN pip3 install -r requirements.txt


COPY test_tf2_image_for_object_detect_ubuntu.py /opt
COPY ssd_mobilenet /opt/ssd_mobilenet
COPY mscoco_label_map.pbtxt /opt
COPY image /opt/image
COPY video.mp4 /opt/

ENV choice=0
ENV img_folder=/opt/image/
ENV vid_file=/opt/video.mp4



WORKDIR /opt/

RUN pip3 install tensorflow
RUN pip3 install opencv-python
RUN pip3 install tensorflow-object-detection-api
RUN pip3 install object-detection

COPY label_map_util.py /usr/local/lib/python3.8/dist-packages/object_detection/utils/

RUN apt install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx 
RUN apt install -y libglib2.0-0
RUN apt install -y libdbus-1-3 libxkbcommon-x11-0 libxcb-icccm4 \
    libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
    libxcb-xinerama0 libxcb-xinput0 libxcb-xfixes0



CMD ["python3","test_tf2_image_for_object_detect_ubuntu.py"]
