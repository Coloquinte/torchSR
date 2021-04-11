#!/bin/sh
directory=./data/DIV2K
urls="
http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip
http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X3.zip
http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip
http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_unknown_X2.zip
http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_unknown_X3.zip
http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_unknown_X4.zip
http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_x8.zip
http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_mild.zip
http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_difficult.zip
http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_wild.zip
http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip
http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X3.zip
http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip
http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_unknown_X2.zip
http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_unknown_X3.zip
http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_unknown_X4.zip
http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_x8.zip
http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_mild.zip
http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_difficult.zip
http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_wild.zip
http://data.vision.ee.ethz.ch/cvl/DIV2K/validation_release/DIV2K_test_LR_bicubic_X2.zip
http://data.vision.ee.ethz.ch/cvl/DIV2K/validation_release/DIV2K_test_LR_bicubic_X3.zip
http://data.vision.ee.ethz.ch/cvl/DIV2K/validation_release/DIV2K_test_LR_bicubic_X4.zip
http://data.vision.ee.ethz.ch/cvl/DIV2K/validation_release/DIV2K_test_LR_unknown_X2.zip
http://data.vision.ee.ethz.ch/cvl/DIV2K/validation_release/DIV2K_test_LR_unknown_X3.zip
http://data.vision.ee.ethz.ch/cvl/DIV2K/validation_release/DIV2K_test_LR_unknown_X4.zip
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
