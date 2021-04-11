#!/bin/sh
directory=./data/REDS
urls="
http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/train_sharp.zip
http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/train_blur.zip
http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/train_blur_comp.zip
http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/train_sharp_bicubic.zip
http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/train_blur_bicubic.zip
http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/val_sharp.zip
http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/val_blur.zip
http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/val_blur_comp.zip
http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/val_sharp_bicubic.zip
http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/val_blur_bicubic.zip
http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/test_blur.zip
http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/test_blur_comp.zip
http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/test_sharp_bicubic.zip
http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/test_blur_bicubic.zip
"

mkdir -p $directory

for url in $urls
do
    wget -P $directory $url
done

for file in $directory/*.zip
do
    unzip $file -d $directory && rm $file
done
