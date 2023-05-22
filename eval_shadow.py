import os
import sys
import tqdm
import ast

import pandas as pd
import numpy as np

from PIL import Image
import cv2
filepath = os.path.split(os.path.abspath(__file__))[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from utils.eval_functions import *
from utils.misc import *

metric_dict = {
    'fm' : Fmeasure(),
    'wfm' : WeightedFmeasure(),
    'sm' : Smeasure(),
    'em' : Emeasure(),
    'mae' : Mae(),
    'mse' : Mse(),
    'mba' : BoundaryAccuracy(),
    'iou' : IoU(),
    'biou' : BIoU(),
    'tiou' : TIoU(),
    'ber' : Ber()
}

def match_scores(opt, args):
    if args.verbose is True:
        print('#' * 20, 'Start Evaluation', '#' * 20)
        datasets = tqdm.tqdm(opt.Eval.datasets, desc='SAM shadow detection ', total=len(
            opt.Eval.datasets), position=0, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
    else:
        datasets = opt.Eval.datasets

    strategies = opt.Eval.strategies
    for strategy in strategies:
        for idx, dataset in enumerate(datasets):
            gt_root = opt.Eval.gt_roots[idx]
            pred_root = os.path.join(opt.Eval.pred_root, dataset)
            preds = os.listdir(pred_root)
            criterion = metric_dict[strategy]

            if args.verbose is True:
                samples = tqdm.tqdm(enumerate(preds), desc=f'strategy: {strategy}, dataset: {dataset}  - match_scores', total=len(
                    preds), position=1, leave=False, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
            else:
                samples = enumerate(preds)

            for i, sample_name in samples:
                txt = os.path.join(pred_root, sample_name, f"{strategy}.txt")
                if os.path.exists(txt):
                    os.remove(txt)
                file = open(txt, "a")
                
                if os.path.exists(os.path.join(pred_root, sample_name, f'{strategies}.txt')):
                    os.remove(os.path.join(pred_root, sample_name, f'{strategies}.txt'))
                
                gt_mask = np.array(Image.open(os.path.join(gt_root, sample_name+'.png')).convert('L'))
                for mask_name in os.listdir(os.path.join(pred_root, sample_name)):
                    if not mask_name.endswith('.png'):
                        continue
                    
                    pred_mask = np.array(Image.open(os.path.join(pred_root, sample_name, mask_name)).convert('L'))

                    if len(pred_mask.shape) != 2:
                        pred_mask = pred_mask[:, :, 0]
                    if len(gt_mask.shape) != 2:
                        gt_mask = gt_mask[:, :, 0]
                        
                    if pred_mask.shape != gt_mask.shape:
                        pred_mask = cv2.resize(pred_mask, gt_mask.shape[::-1])

                    assert pred_mask.shape == gt_mask.shape, print(pred_mask.shape, 'does not match the size of', gt_mask.shape)
                    
                    criterion.step(pred=pred_mask, gt=gt_mask, file=file, name=mask_name)

                file.close()


def evaluate(opt, args):
    if os.path.isdir(opt.Eval.result_path) is False:
        os.makedirs(opt.Eval.result_path)

    if args.verbose is True:
        print('#' * 20, 'Start Evaluation', '#' * 20)
        datasets = tqdm.tqdm(opt.Eval.datasets, desc='Evaluation', total=len(
            opt.Eval.datasets), position=0, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
    else:
        datasets = opt.Eval.datasets

    # create result file
    result_path = './shadow_result.txt'
    if os.path.exists(result_path):
        os.remove(result_path)
    result_file = open(result_path, 'a')
    strategies = opt.Eval.strategies
    
    for strategy in strategies:
        results = []
        for idx, dataset in enumerate(datasets):
            os.makedirs(os.path.join(opt.Eval.result_path, dataset), exist_ok=True)
            gt_root = opt.Eval.gt_roots[idx]
            
            pred_root = os.path.join(opt.Eval.pred_root, dataset)
            preds = os.listdir(pred_root)

            BER = Ber()

            print('.'*30, len(preds),)
            if args.verbose is True:
                samples = tqdm.tqdm(enumerate(preds), desc=dataset + ' - Evaluation', total=len(
                    preds), position=1, leave=False, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
            else:
                samples = tqdm.tqdm(enumerate(preds))

            for i, sample_name in samples:
                txt = os.path.join(pred_root, sample_name, f'{strategy}.txt')
                            
                mx, pred = 0, None    
                with open(txt, 'r') as file:
                    line = file.readline()
                    while line:
                        f = float(line.split(' ')[-1])
                        if f > mx:
                            mx, pred= f, line.split(' ')[0]
                        line = file.readline()
                            
                file.close()
                
                pred_mask = np.array(Image.open(os.path.join(pred_root, sample_name, pred)).convert('L'))
                gt_mask = np.array(Image.open(os.path.join(gt_root, sample_name+'.png')).convert('L'))

                # save
                Image.fromarray(pred_mask).save(os.path.join(f'./pred/{dataset}', f'{strategy}_{pred[:-4]}_'+sample_name+'.png'))
                if sample_name in ['rot-blau-bunt glatt.png', 'shadow__s_car_found_by_blazerona-d5esxqc.png']:
                    print('--> ', pred, os.path.join(f'./pred/{dataset}', f'{strategy}_{pred[:-4]}_'+sample_name+'.png'))

                if len(pred_mask.shape) != 2:
                    pred_mask = pred_mask[:, :, 0]
                if len(gt_mask.shape) != 2:
                    gt_mask = gt_mask[:, :, 0]
                    
                if pred_mask.shape != gt_mask.shape:
                    pred_mask = cv2.resize(pred_mask, gt_mask.shape[::-1])

                BER.step(pred=pred_mask, gt=gt_mask)
                
            result = []
            ber = BER.get_results()
            print(f'{strategy}, {dataset}\r\n', ber)
            # dump to file
            result_file.write(f'strategy: {strategy}, dataset: {dataset}\r\n')
            for key, value in ber.items():
                result_file.write(f'{key}: {value}' + '\r\n')
            result_file.write('\r\n')
            result_file.flush()

        if args.verbose is True:
            for dataset, result in zip(datasets, results):
                print('###', dataset, '###', '\n', result.sort_index(), '\n')

    result_file.close()


if __name__ == "__main__":
    args = parse_args()
    opt = load_config(args.config)
    # match_scores(opt, args)
    evaluate(opt, args)