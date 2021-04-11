#!/bin/sh
directory=./data/SRBenchmarks
urls="
http://cv.snu.ac.kr/research/EDSR/benchmark.tar
"

mkdir -p $directory

for url in $urls
do
    wget -P $directory $url
done

for file in $directory/benchmark.tar
do
    tar -xf $file -C $directory && rm $file
done

