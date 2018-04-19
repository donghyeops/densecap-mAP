#-*- coding: utf-8 -*-

# 구현 대상
#   https://github.com/jcjohnson/densecap/tree/master/eval
# 참고
#   https://sites.google.com/site/hyunguk1986/personal-study/-ap-map-recall-precision
from metric.meteor import Meteor
import numpy as np
from ..utils.cython_bbox import bbox_overlaps
from ..datasets.visual_genome_loader import visual_genome


# Usage : call evaluate_caption(vg, gt_caption, gt_region, pred_caption, pred_region)
# requirement : pytorch, java 1.8.0(for Meteor), cython, numpy ...
# Dataset : Visual Genome dataset

class Caption_Evaluator():
    def __init__(self, base_metric='meteor', final_metric='mAP', thr_ious=None, thr_scores=None):
        self.base_metric = base_metric
        self.final_metric = final_metric
        
        if base_metric == 'meteor': # 현재는 Meteor만 측정
            self.Evaluator = Meteor()
            
        if thr_ious == None:
            self.thr_ious = [0.3, 0.4, 0.5, 0.6, 0.7] # mAP 측정 시, 고려하는 iou 범위
        else:
            self.thr_ious = thr_ious
            
        if thr_scores == None:
            self.thr_scores = [0, 0.05, 0.1, 0.15, 0.2, 0.25] # mAP 측정 시, 고려하는 스코어 범위
        else:
            self.thr_scores = thr_scores
    
    # 모델로부터 예측된 결과와 정답을 비교하여 점수를 평가 (기본: Meteor:mAP)
    # vg : DATASET 클래스. index to word를 위함
    # gt_caption : 정답 캡션 리스트 (word들의 index로 구성) [gt#, max_len]
    # gt_region : 정답 캡션 영역 리스트 (각 캡션에 해당되는 bounding box) [gt#, 4(x1, y2, x2, y2)]
    # pred_caption : 예측된 캡션 리스트 (word들의 index로 구성) [pred#, max_len]
    # pred_region : 예측된 캡션 영역 리스트 (각 캡션에 해당되는 bounding box) [pred#, 4(x1, y2, x2, y2)]
    def evaluate_caption(self, vg, gt_caption, gt_region, pred_caption, pred_region):
        IoUs = self.__get_IoUs(gt_region, pred_region)
        
        if self.final_metric == 'mAP':
            score = self.__get_mAP(vg, gt_caption, pred_caption, IoUs)
        else:
            raise Exception('None Score :', self.final_metric)
        
        return score
        
    # gt와 예측된 region간의 iou를 계산
    def __get_IoUs(self, gt_region, pred_region):
        # overlaps (pred#, gt#)
        overlaps = bbox_overlaps(
            np.ascontiguousarray(pred_region, dtype=np.float),
            np.ascontiguousarray(gt_region, dtype=np.float))
        
        #return IoUs # ndarry (pred#, gt#) # 최대값만 실수, 나머진 0
        return overlaps
    
    # 두 캡션 리스트로부터 mAP를 구함
    def __get_mAP(self, vg, gt_caption, pred_caption, IoUs):
        # precision_list = {'0.3_0':0., '0.3_0.05':0., ... } # [mAP]
        AP_dict = {str(score)+'_'+str(iou):0. for score in self.thr_scores for iou in self.thr_ious}
        
        gt_caption = gt_caption.astype(int) # np.float32 -> int
        pred_caption = pred_caption.numpy().astype(int)
        
        gt_cnt = len(gt_caption)
        gt_t_caps = [vg.untokenize_single_sentence(cap) for cap in gt_caption]
        pred_t_caps = [vg.untokenize_single_sentence(cap) for cap in pred_caption]
        pred_t_caps2 = []
        for cap in pred_t_caps:
            if not len(cap) == 0:
                pred_t_caps2.append(cap)
        
        gt_num = len(gt_t_caps)
        pred_num = len(pred_t_caps2)
        
        scores = np.zeros((pred_num, gt_num)) # score 계산 값을 저장해둠
        for idx in range(pred_num): # pred# 만큼 반복, iou는 gt#에 대한 iou리스트
            for t_idx in range(gt_num):
                result, _ = self.Evaluator.compute_score({0:[gt_t_caps[t_idx]]}, {0:[pred_t_caps2[idx]]})
                scores[idx, t_idx] = result
                
        # threshold 별 AP 계산
        for scr_thr in self.thr_scores:
            for iou_thr in self.thr_ious:
                correct_cnt = 0
                precision_list = []
                for idx in range(pred_num): # pred# 만큼 반복
                    for t_idx in range(gt_num):
                        if IoUs[idx][t_idx] < iou_thr:
                            continue
                        if scores[idx, t_idx] >= scr_thr:
                            correct_cnt += 1
                            precision_list.append(correct_cnt / (idx+1))
                            break
                    
                AP_dict[str(scr_thr)+'_'+str(iou_thr)] = np.average(precision_list) if len(precision_list) > 0 else 0
        sum_AP = 0.
        for key in AP_dict.keys(): # 모든 threshold 조합에 대한 ap 합
            sum_AP += AP_dict[key]
        mAP = sum_AP / len(AP_dict.keys()) # 전체 mAP 평균
        #print '******', mAP, '******' ## test
        return mAP # double
        

