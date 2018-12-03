# Usage Pipeline for Object Detection
Since we are using the tensorflow object detection API to train our model, some code needs to be executed within its directories because we call on their scripts, and some of their scripts rely on the caller to be in certain directories when making relative references to other modules.
## Pipeline instructions, details in each file:
1. Clone the [tensorflow models repository](https://github.com/tensorflow/models.git) into the main directory.
1. Go to **models/research** and run **setup.py build** followed by **setup.py install**.
1. In the same directory, run **protoc_script.sh**.
2. Assuming there is a video file called **pokemon.mp4**, run **extract_frames.py**.
3. (Optional) Assuming there is a training dataset, and ImageMagick is installed, **run mogrify_script.sh** to avoid getting warning messages in the following step.
4. Assuming there is a training dataset, run **generate_tf_record.py**.
5. Under **models/**, create a directory called **model** and add **pipeline.config** into that directory.
6. From **models/research** run **train.sh**, you may get an error _can't pickle dict values_ if you are using python 3 during evaluation
7. In **export.sh**, change the number in **model.ckpt-###** to the last checkpoint number produced in training under **models/model**, and run it from **models/research**
8. Run **detection.py** from **models/research/object_detection**, the results should be stored in a folder called **detected_results**
