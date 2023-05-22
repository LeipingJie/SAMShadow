#! /bin/bash

python amg.py --checkpoint ./sam_vit_h_4b8939.pth --input /dataset/shadow/SBU-shadow/SBU-Test/images --output ./preds/sbu
python amg.py --checkpoint ./sam_vit_h_4b8939.pth --input /dataset/shadow/UCF/InputImages --output ./preds/ucf
python amg.py --checkpoint ./sam_vit_h_4b8939.pth --input /dataset/shadow/ISTD_Dataset/test/test_A --output ./preds/istd
python amg.py --checkpoint ./sam_vit_h_4b8939.pth --input /dataset/shadow/merged_cuhk/image --output ./preds/cuhk