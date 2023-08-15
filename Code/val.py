import argparse
import json
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
from scripts import test_hashing

parser = argparse.ArgumentParser()
parser.add_argument('-l', default='/home/zf/orthohash/logs/alexnet128_utkface_multicls2_1_100_0.0001_adam_1.0/orthohash_42_071/', help='training logdir')
parser.add_argument('-m', type=float, default=0, help='threshold value for ternary')

args = parser.parse_args()

logdir = args.l
config = json.load(open(logdir + '/config.json'))

config.update({
    'map_threshold': args.m
})

test_hashing.main(config)
