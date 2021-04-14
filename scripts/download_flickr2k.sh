#!/bin/sh
directory=./data/Flickr2K
urls="
https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar
"

mkdir -p $directory

for url in $urls
do
    wget -P $directory $url
done

for file in $directory/Flickr2K.tar
do
    tar -xf $file -C $directory && rm $file
done

