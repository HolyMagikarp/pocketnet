import tensorflow as tf
import sys
import os
sys.path.insert(0, r"{}\models\research\object_detection\utils".format(os.getcwd()))
import dataset_util
import cv2

# NEED:
#    Protoc
#    models repo https://github.com/tensorflow/models.git
# Steps to do before running:
#    1. Clone models repository to the current directory
#    2. Run protoc_script.sh
#    3. Run models/research/setup.py with argument build

flags = tf.app.flags
flags.DEFINE_string('out_path', 'TFRecords/test.record', "Path for output TFRecord")
FLAGS = flags.FLAGS

def create_tf_example(img_path, class_text, class_label):
    """Create a tf.Example proto from input img
    Args:
        img_path: path to the image
        class_text: string of the class
        class_label: int label of the class
        
    Return:
        tf.Example object
    """
    img_mat = cv2.imread(img_path)
    imgFile = open(img_path, "rb")
    img = imgFile.read()
    
    filename = img_path.encode('utf-8')
    img_format = b'img_path.split(".")[-1]'
    
    height, width, _ = img_mat.shape
    
    xmin = [0.]
    xmax = [1.]
    ymin = [0.]
    ymax = [1.]
    classes_text = [class_text.encode('utf-8')]
    classes = [class_label]
    
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(img),
        'image/format': dataset_util.bytes_feature(img_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    
    return tf_example

# you may get warnings like "libpng warning: iCCP: known incorrect sRGB profile"
# when running this, it doesn't matter
# if you don't want to see the warnings then install ImageMagick, check the 
# legacy option as an installed feature, then run the mogrify_script.sh

# opencv doesn't read gif files and I'm too lazy to convert them automatically
# so if u have gif in your dataset either convert or delete
def main(_):
    # directory names under /dataset/
    # the training images should in be in these directories
    pokemons = ["pikachu", "charmander", "gastly", "gengar", "haunter", "meowth"]
    
    os.makedirs("TFRecords", exist_ok=True)
    writer = tf.python_io.TFRecordWriter(FLAGS.out_path)
    
    for i in range(len(pokemons)):
        pokemon = pokemons[i]
        dir = "dataset/{}/".format(pokemon)
        for img_name in os.listdir(dir):
            # class label is i + 1 because it starts at 1
            tf_example = create_tf_example(dir + img_name, pokemons[i], i + 1)
            writer.write(tf_example.SerializeToString())
    writer.close()


if __name__ == "__main__":
    tf.app.run()
