nvidia-docker run -it --shm-size 314g --rm --network host \
-v <HOST DATA FOLDER>:<CONTAINER DATA FOLDER> \
-v <HOST SRC CODE FOLDER>:<CONTAINER SRC CODE FOLDER> \
 ufoym/deepo:all-py36-cu111

