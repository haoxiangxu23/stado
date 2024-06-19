#! /bin/bash

root_path=$(cd $(dirname $0) && pwd)/..
data_path=$root_path/data
src_path=$root_path/src
model_path_rgb=$root_path/experiment/result_model/hmdb_s1_dla34_K7_rgb_coco.pth
model_path_flow=$root_path/experiment/result_model/hmdb_s1_dla34_K7_flow_coco.pth
result_path_rgb=$root_path/result/inference/train00_occXX_rgb
result_path_flow=$root_path/result/inference/train00_occXX_flow
result_path_both=$root_path/result/inference/train00_occXX_both
origin_data="JHMDB_XX"
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
mkdir -p $result_path_rgb
mkdir -p $result_path_flow
mkdir -p $result_path_both

# --------------- rgb ----------------
# inference
python3 det.py --task normal --dataset hmdb --split 1 --hm_fusion_rgb 0.4 \
--batch_size 8 --master_batch 8 --num_workers 4 --gpus 0 --flip_test \
--ninput 1 --rgb_model $model_path_rgb --inference_dir $result_path_rgb
# frame
python3 ACT.py --task frameAP --th 0.5 --dataset hmdb --split 1 --inference_dir $result_path_rgb
# vidio
python3 ACT.py --task BuildTubes --dataset hmdb --split 1 --inference_dir $result_path_rgb
python3 ACT.py --task videoAP --th 0.2 --dataset hmdb --split 1 --inference_dir $result_path_rgb
python3 ACT.py --task videoAP --th 0.5 --dataset hmdb --split 1 --inference_dir $result_path_rgb
python3 ACT.py --task videoAP --th 0.75 --dataset hmdb --split 1 --inference_dir $result_path_rgb
python3 ACT.py --task videoAP_all --dataset hmdb --split 1 --inference_dir $result_path_rgb

# -------------- flow ----------------
# inference
python3 det.py --task normal --dataset hmdb --split 1 --hm_fusion_rgb 0.4 \
--batch_size 8 --master_batch 8 --num_workers 4 --gpus 0 --flip_test \
--ninput 5 --flow_model $model_path_flow --inference_dir $result_path_flow
# frame
python3 ACT.py --task frameAP --th 0.5 --dataset hmdb --split 1 --inference_dir $result_path_flow
# vidio
python3 ACT.py --task BuildTubes --dataset hmdb --split 1 --inference_dir $result_path_flow
python3 ACT.py --task videoAP --th 0.2 --dataset hmdb --split 1 --inference_dir $result_path_flow
python3 ACT.py --task videoAP --th 0.5 --dataset hmdb --split 1 --inference_dir $result_path_flow
python3 ACT.py --task videoAP --th 0.75 --dataset hmdb --split 1 --inference_dir $result_path_flow
python3 ACT.py --task videoAP_all --dataset hmdb --split 1 --inference_dir $result_path_flow

# ------------ rgb+flow --------------
# inference
python3 det.py --task normal --dataset hmdb --split 1 --hm_fusion_rgb 0.4 \
--batch_size 8 --master_batch 8 --num_workers 4 --gpus 0 --flip_test \
--ninput 5 --rgb_model $model_path_rgb --flow_model $model_path_flow --inference_dir $result_path_both
# fream 
python3 ACT.py --task frameAP --th 0.5 --dataset hmdb --split 1 --inference_dir $result_path_both
# vidio
python3 ACT.py --task BuildTubes --dataset hmdb --split 1 --inference_dir $result_path_both
python3 ACT.py --task videoAP --th 0.2 --dataset hmdb --split 1 --inference_dir $result_path_both
python3 ACT.py --task videoAP --th 0.5 --dataset hmdb --split 1 --inference_dir $result_path_both
python3 ACT.py --task videoAP --th 0.75 --dataset hmdb --split 1 --inference_dir $result_path_both
python3 ACT.py --task videoAP_all --dataset hmdb --split 1 --inference_dir $result_path_both

cd $data_path
mv "$target_data" "$origin_data"
exit 0