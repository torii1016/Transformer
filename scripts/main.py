import argparse
import os
import numpy as np

import tensorflow as tf

from model.simple_attention import SimpleAttention

def main(config_dict):

    simple_attention = SimpleAttention()

if __name__ == '__main__':
    parser = argparse.ArgumentParser( description='Process some integers' )
    parser.add_argument( '--config', default="config/config.toml", type=str, help="default: config/config.toml")
    args = parser.parse_args()

    main(None)