#! /bin/bash

root_path=$(cd $(dirname $0) && pwd)/..
data_path=$root_path/data
src_path=$root_path/src
save_dir=$root_path/result/train/verX_occ25/rgb
origin_data="JHMDB_25"
target_data="JHMDB"

cd $data_path
if [ ! -d "$origin_data" ]; then
  echo "[ERROR] $origin_data does not exist"
  exit 1
fi
if [ -d "$target_data" ]; then
  echo "[ERROR] $target_data exists"
  exit 1
fi
mv "$origin_data" "$target_data"

cd $src_path
mkdir -p $save_dir

python3 train.py --dataset hmdb --split 1 --lr 5e-4 --lr_step 10,23 --ninput 1 \
--batch_size 8 --master_batch 8 --num_workers 4 --gpus 0 --num_epochs 30 \
--auto_stop --rgb_model $save_dir --mgan_weight 5

cd $data_path
mv "$target_data" "$origin_data"
exit 0