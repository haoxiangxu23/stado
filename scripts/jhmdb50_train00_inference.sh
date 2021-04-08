cd data
mv JHMDB_50 JHMDB
cd ..
mkdir -p occ_inference_results/occ50_train00_rgb
mkdir -p occ_inference_results/occ50_train00_flow
mkdir -p occ_inference_results/occ50_train00_both
cd src

# --------------- rgb ----------------
# inference
python3 det.py --task normal --K 7 --gpus 0 --dataset hmdb --split 1 \
--batch_size 8 --master_batch 8 --num_workers 4 \
--flip_test --ninput 5 --hm_fusion_rgb 0.4 \
--rgb_model ../models/hmdb_s1_dla34_K7_rgb_coco.pth \
--inference_dir ../occ_inference_results/occ50_train00_rgb/
# fream 
python3 ACT.py --task frameAP --K 7 --th 0.5 --dataset hmdb --split 1 \
--inference_dir ../occ_inference_results/occ50_train00_rgb/
# vidio
python3 ACT.py --task BuildTubes --K 7 --dataset hmdb --split 1 \
--inference_dir ../occ_inference_results/occ50_train00_rgb/
python3 ACT.py --task videoAP --K 7 --th 0.2 --dataset hmdb --split 1 \
--inference_dir ../occ_inference_results/occ50_train00_rgb/
python3 ACT.py --task videoAP --K 7 --th 0.5 --dataset hmdb --split 1 \
--inference_dir ../occ_inference_results/occ50_train00_rgb/
python3 ACT.py --task videoAP --K 7 --th 0.75 --dataset hmdb --split 1 \
--inference_dir ../occ_inference_results/occ50_train00_rgb/
python3 ACT.py --task videoAP_all --K 7 --dataset hmdb --split 1 \
--inference_dir ../occ_inference_results/occ50_train00_rgb/

# -------------- flow ----------------
# inference
python3 det.py --task normal --K 7 --gpus 0 --dataset hmdb --split 1 \
--batch_size 8 --master_batch 8 --num_workers 4 \
--flip_test --ninput 5 --hm_fusion_rgb 0.4 \
--flow_model ../models/hmdb_s1_dla34_K7_flow_coco.pth \
--inference_dir ../occ_inference_results/occ50_train00_flow/
# fream 
python3 ACT.py --task frameAP --K 7 --th 0.5 --dataset hmdb --split 1 \
--inference_dir ../occ_inference_results/occ50_train00_flow/
# vidio
python3 ACT.py --task BuildTubes --K 7 --dataset hmdb --split 1 \
--inference_dir ../occ_inference_results/occ50_train00_flow/
python3 ACT.py --task videoAP --K 7 --th 0.2 --dataset hmdb --split 1 \
--inference_dir ../occ_inference_results/occ50_train00_flow/
python3 ACT.py --task videoAP --K 7 --th 0.5 --dataset hmdb --split 1 \
--inference_dir ../occ_inference_results/occ50_train00_flow/
python3 ACT.py --task videoAP --K 7 --th 0.75 --dataset hmdb --split 1 \
--inference_dir ../occ_inference_results/occ50_train00_flow/
python3 ACT.py --task videoAP_all --K 7 --dataset hmdb --split 1 \
--inference_dir ../occ_inference_results/occ50_train00_flow/

# ------------ rgb+flow --------------
# inference
python3 det.py --task normal --K 7 --gpus 0 --dataset hmdb --split 1 \
--batch_size 8 --master_batch 8 --num_workers 4 \
--flip_test --ninput 5 --hm_fusion_rgb 0.4 \
--rgb_model ../models/hmdb_s1_dla34_K7_rgb_coco.pth \
--flow_model ../models/hmdb_s1_dla34_K7_flow_coco.pth \
--inference_dir ../occ_inference_results/occ50_train00_both/
# fream 
python3 ACT.py --task frameAP --K 7 --th 0.5 --dataset hmdb --split 1 \
--inference_dir ../occ_inference_results/occ50_train00_both/
# vidio
python3 ACT.py --task BuildTubes --K 7 --dataset hmdb --split 1 \
--inference_dir ../occ_inference_results/occ50_train00_both/
python3 ACT.py --task videoAP --K 7 --th 0.2 --dataset hmdb --split 1 \
--inference_dir ../occ_inference_results/occ50_train00_both/
python3 ACT.py --task videoAP --K 7 --th 0.5 --dataset hmdb --split 1 \
--inference_dir ../occ_inference_results/occ50_train00_both/
python3 ACT.py --task videoAP --K 7 --th 0.75 --dataset hmdb --split 1 \
--inference_dir ../occ_inference_results/occ50_train00_both/
python3 ACT.py --task videoAP_all --K 7 --dataset hmdb --split 1 \
--inference_dir ../occ_inference_results/occ50_train00_both/

cd ..