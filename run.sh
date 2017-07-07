#!/bin/sh
ROOT=/media/oukohou/E/gong_an_pictures0612/gong_an_pictures

export PYTHONPATH=$ROOT
echo $@
python $ROOT/inference/inference_single_image.py $@
