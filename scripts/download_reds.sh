#!/bin/sh
mkdir -p ./data/REDS
wget -P ./data/REDS http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/train_sharp.zip
wget -P ./data/REDS http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/train_blur.zip
wget -P ./data/REDS http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/train_blur_comp.zip
wget -P ./data/REDS http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/train_sharp_bicubic.zip
wget -P ./data/REDS http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/train_blur_bicubic.zip
wget -P ./data/REDS http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/val_sharp.zip
wget -P ./data/REDS http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/val_blur.zip
wget -P ./data/REDS http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/val_blur_comp.zip
wget -P ./data/REDS http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/val_sharp_bicubic.zip
wget -P ./data/REDS http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/val_blur_bicubic.zip
wget -P ./data/REDS http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/test_blur.zip
wget -P ./data/REDS http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/test_blur_comp.zip
wget -P ./data/REDS http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/test_sharp_bicubic.zip
wget -P ./data/REDS http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/test_blur_bicubic.zip
