mkdir -p ./data/DIV2K
# Training datasets
wget -P ./data/DIV2K http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
wget -P ./data/DIV2K http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip
wget -P ./data/DIV2K http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X3.zip
wget -P ./data/DIV2K http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip
wget -P ./data/DIV2K http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_unknown_X2.zip
wget -P ./data/DIV2K http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_unknown_X3.zip
wget -P ./data/DIV2K http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_unknown_X4.zip
wget -P ./data/DIV2K http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_x8.zip
wget -P ./data/DIV2K http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_mild.zip
wget -P ./data/DIV2K http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_difficult.zip
wget -P ./data/DIV2K http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_wild.zip
# Validation datasets
wget -P ./data/DIV2K http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
wget -P ./data/DIV2K http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip
wget -P ./data/DIV2K http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X3.zip
wget -P ./data/DIV2K http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip
wget -P ./data/DIV2K http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_unknown_X2.zip
wget -P ./data/DIV2K http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_unknown_X3.zip
wget -P ./data/DIV2K http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_unknown_X4.zip
wget -P ./data/DIV2K http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_x8.zip
wget -P ./data/DIV2K http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_mild.zip
wget -P ./data/DIV2K http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_difficult.zip
wget -P ./data/DIV2K http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_wild.zip
# Testing datasets; no HR, and not all tracks
wget -P ./data/DIV2K http://data.vision.ee.ethz.ch/cvl/DIV2K/validation_release/DIV2K_test_LR_bicubic_X2.zip
wget -P ./data/DIV2K http://data.vision.ee.ethz.ch/cvl/DIV2K/validation_release/DIV2K_test_LR_bicubic_X3.zip
wget -P ./data/DIV2K http://data.vision.ee.ethz.ch/cvl/DIV2K/validation_release/DIV2K_test_LR_bicubic_X4.zip
wget -P ./data/DIV2K http://data.vision.ee.ethz.ch/cvl/DIV2K/validation_release/DIV2K_test_LR_unknown_X2.zip
wget -P ./data/DIV2K http://data.vision.ee.ethz.ch/cvl/DIV2K/validation_release/DIV2K_test_LR_unknown_X3.zip
wget -P ./data/DIV2K http://data.vision.ee.ethz.ch/cvl/DIV2K/validation_release/DIV2K_test_LR_unknown_X4.zip

mkdir -p ./data/SRBenchmarks
# Benchmark datasets (Set5, Set14, B100, Urban100)
wget -P ./data/SRBenchmarks https://cv.snu.ac.kr/research/EDSR/benchmark.tar
