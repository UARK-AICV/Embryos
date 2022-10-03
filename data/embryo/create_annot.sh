DATA_NAME='human'
IMG_DIR='dataset/human/images'
FEAT_SAVE_DIR='dataset/human/resnet_fold0' #.npy files


# ------------------------------------------------------------
CSV_PATH='dataset/human/embryo_train_0.csv'
SAVE_PATH='dataset/human/train_0.json'
python data/embryo/create_annot.py $DATA_NAME \
$CSV_PATH $IMG_DIR $FEAT_SAVE_DIR $SAVE_PATH

# ------------------------------------------------------------
CSV_PATH='dataset/human/embryo_val_0.csv'
SAVE_PATH='dataset/human/val_0.json'
python data/embryo/create_annot.py $DATA_NAME \
$CSV_PATH $IMG_DIR $FEAT_SAVE_DIR $SAVE_PATH

# ------------------------------------------------------------
CSV_PATH='dataset/human/embryo_test_0.csv'
SAVE_PATH='dataset/human/test_0.json'
python data/embryo/create_annot.py $DATA_NAME \
$CSV_PATH $IMG_DIR $FEAT_SAVE_DIR $SAVE_PATH 