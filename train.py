import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

import utils
from models import DMCNN
import os

flags = tf.flags
flags.DEFINE_string("gpu", "0", "The GPU to run on")
flags.DEFINE_string("mode", "HMEAE", "DMCNN or HMEAE")
flags.DEFINE_string("classify", "tuple", "single or tuple")

def main(_):
    config = flags.FLAGS
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    extractor = utils.Extractor()
    extractor.Extract()
    loader = utils.Loader()
    t_data = loader.load_trigger()
    a_data = loader.load_argument()
    trigger = DMCNN(t_data,a_data,loader.maxlen,loader.max_argument_len,loader.wordemb)
    a_data_process = trigger.train_trigger()
    argument = DMCNN(t_data,a_data_process,loader.maxlen,loader.max_argument_len,loader.wordemb,stage=config.mode,classify=config.classify)
    argument.train_argument()

if __name__=="__main__":
    #动态分配内存
    session = tf.Session(config=config)
    tf.app.run()