import numpy as np
import pandas as pd
import scipy


################################## Instance detection evaluation: get matched gts and preds as well as iou, precision and recall matrices
def get_detections(instance_labels, instance_preds, min_iou_match, non_tree_label):
    iou_matrix = np.zeros((np.max(instance_preds)+1, np.max(instance_labels)+1))
    precision_matrix = np.zeros((np.max(instance_preds)+1, np.max(instance_labels)+1))
    recall_matrix = np.zeros((np.max(instance_preds)+1, np.max(instance_labels)+1))
    
    ################## calculate iou, precision and recall for each instance_pred and instance_label
    for instance_pred in np.arange(np.max(instance_preds)+1):
        instance_pred_mask = instance_preds == instance_pred
        instance_labels_part_of_instance_pred = np.unique(instance_labels[instance_pred_mask])
        instance_labels_part_of_instance_pred = instance_labels_part_of_instance_pred[instance_labels_part_of_instance_pred != non_tree_label]

        for instance_label_part_of_instance_pred in instance_labels_part_of_instance_pred:
            instance_label_mask = instance_labels == instance_label_part_of_instance_pred

            tp, fp, tn, fn = get_eval_components(instance_pred_mask, instance_label_mask)
            prec, rec, iou = get_segmentation_metrics(tp, fp, fn)
            iou_matrix[int(instance_pred), int(instance_label_part_of_instance_pred)] = iou
            precision_matrix[int(instance_pred), int(instance_label_part_of_instance_pred)] = prec
            recall_matrix[int(instance_pred), int(instance_label_part_of_instance_pred)] = rec

    ################## Perform hungarian matching between instance_preds and instance_labels based on iou_matrix and filter matches based on minimum iou
    matched_preds_preliminary, matched_gts_preliminary = scipy.optimize.linear_sum_assignment(iou_matrix, maximize=True)
    mask_satisfies_match_condition = iou_matrix[matched_preds_preliminary, matched_gts_preliminary] > min_iou_match
    matched_preds, matched_gts = matched_preds_preliminary[mask_satisfies_match_condition], matched_gts_preliminary[mask_satisfies_match_condition]
    return matched_gts, matched_preds, iou_matrix, precision_matrix, recall_matrix


################################## Instance detection evaluation: get non matched gts and preds as well as additional info them for visualization of errors
def get_detection_failures(matched_gts, matched_preds, unique_instance_labels, unique_instance_preds, iou_matrix, 
                           precision_matrix, recall_matrix, min_precision_for_pred, min_recall_for_gt):
    ################## get non matched preds and gts
    assert (iou_matrix[matched_preds, matched_gts] > 0).sum() == len(matched_preds), 'a zero iou correspondence has been matched'
    non_matched_preds = np.array(list(set(unique_instance_preds) - set(matched_preds))).astype(np.int64)
    non_matched_gts = np.array(list(set(unique_instance_labels) - set(matched_gts))).astype(np.int64)

    ################## get additional info for non matched preds
    non_matched_preds_corresponding_gt = []
    for non_matched_pred in non_matched_preds:
        # at least min_precision_for_pred percent of a non-matched pred should belong to gts in order to be considered as a commission error.
        # Otherwise it might be a detection of an unlabeled tree or bush, which is not counted as a commission error.
        if precision_matrix[non_matched_pred].sum() < min_precision_for_pred:
            non_matched_preds_corresponding_gt.append(np.nan)
        else:
            non_matched_preds_corresponding_gt.append(precision_matrix[non_matched_pred].argmax())
    non_matched_preds_corresponding_gt = np.array(non_matched_preds_corresponding_gt)

    ################## get additional info for non matched gts
    # If there is a prediction that has a sufficiently high recall (> min_recall_for_gt) with the non_matched_gt, we consider this a case of undersegmentation.
    # In this case, this part of the code identifies the prediction that represents the undersegmented trees (non_matched_gts_corresponding_pred),
    # and the matched ground truth tree that corresponds to this prediction (non_matched_gts_corresponding_other_tree).
    non_matched_gts_corresponding_pred = []
    non_matched_gts_corresponding_other_tree = []
    for non_matched_gt in non_matched_gts:
        if recall_matrix[:, non_matched_gt].max() < min_recall_for_gt: # no undersegmentation error, e.g. in case that the tree is not detected at all
            non_matched_gts_corresponding_other_tree.append(np.nan)
            non_matched_gts_corresponding_pred.append(np.nan)
        else: # identify corresponding pred and other tree with highest recall > min_recall_for_gt (if it exists)
            corresponding_pred = np.argmax(recall_matrix[:, non_matched_gt])
            non_matched_gts_corresponding_pred.append(corresponding_pred)
            other_gts = np.delete(np.arange(recall_matrix.shape[1]), non_matched_gt)
            argmin_recall_other_gts = recall_matrix[corresponding_pred, other_gts].argmax()
            
            if recall_matrix[corresponding_pred, other_gts][argmin_recall_other_gts] < min_recall_for_gt:
                non_matched_gts_corresponding_other_tree.append(np.nan)
            else:
                non_matched_gts_corresponding_other_tree.append(other_gts[argmin_recall_other_gts])
                
    non_matched_gts_corresponding_pred = np.array(non_matched_gts_corresponding_pred)
    non_matched_gts_corresponding_other_tree = np.array(non_matched_gts_corresponding_other_tree)
    return non_matched_gts, non_matched_preds, non_matched_preds_corresponding_gt, non_matched_gts_corresponding_pred, non_matched_gts_corresponding_other_tree


################################## Instance segmentation evaluation
def evaluate_instance_segmentation(instance_preds, instance_labels, unique_gts, unique_preds, coords, 
                                   mapping_to_original_gt_nums, mapping_to_original_pred_nums,
                                   xy_partition, z_partition):
    
    no_partition = evaluate_no_partition(instance_preds, instance_labels, unique_gts, unique_preds, 
                                         mapping_to_original_gt_nums, mapping_to_original_pred_nums)
    if xy_partition:
        xy = evaluate_xy_partition(instance_preds, instance_labels, unique_gts, unique_preds, coords, xy_partition,
                                            mapping_to_original_gt_nums, mapping_to_original_pred_nums)
    else:
        xy = None
    if z_partition:
        z = evaluate_z_partition(instance_preds, instance_labels, unique_gts, unique_preds, coords, z_partition,
                                          mapping_to_original_gt_nums, mapping_to_original_pred_nums)
    else:
        z = None
    return no_partition, xy, z


################################## Instance segmentation evaluation for complete trees (no partition)
def evaluate_no_partition(instance_preds, instance_labels, unique_gts, unique_preds, mapping_to_original_gt_nums, mapping_to_original_pred_nums):
    val_res = dict()
    val_res['instance_pred'] = []
    val_res['instance_label'] = []
    val_res['prec'] = []
    val_res['rec'] = []
    val_res['iou'] = []

    # get results
    for instance_pred, instance_label in zip(unique_preds, unique_gts):
        val_res['instance_pred'].append(mapping_to_original_pred_nums[instance_pred])
        val_res['instance_label'].append(mapping_to_original_gt_nums[instance_label])

        ind_pred_positive = instance_preds == instance_pred
        ind_positive = instance_labels == instance_label

        tp, fp, tn, fn = get_eval_components(ind_pred_positive, ind_positive)
        prec, rec, iou = get_segmentation_metrics(tp, fp, fn)
        val_res['prec'].append(prec)
        val_res['rec'].append(rec)
        val_res['iou'].append(iou)
    
    val_res_df = pd.DataFrame.from_dict(val_res)
    return val_res_df


################################## Instance segmentation evaluation for the xy partition of trees
def evaluate_xy_partition(instance_preds, instance_labels, unique_gts, unique_preds, coords, intvls, mapping_to_original_gt_nums, mapping_to_original_pred_nums):
    val_res = dict()
    val_res['instance_pred'] = []
    val_res['instance_label'] = []
    for i in range(len(intvls) - 1):
        val_res[f'prec_intvl{intvls[i]}_{intvls[i+1]}'] = []
    for i in range(len(intvls) - 1):
        val_res[f'rec_intvl{intvls[i]}_{intvls[i+1]}'] = []
    for i in range(len(intvls) - 1):
        val_res[f'iou_intvl{intvls[i]}_{intvls[i+1]}'] = []

    # get results
    for instance_pred, instance_label in zip(unique_preds, unique_gts):
        val_res['instance_pred'].append(mapping_to_original_pred_nums[instance_pred])
        val_res['instance_label'].append(mapping_to_original_gt_nums[instance_label])

        ind_pred_positive = instance_preds == instance_pred
        ind_positive = instance_labels == instance_label
    
        # calculate tree position and center xy-coordinates according to it
        tree_coords = coords[ind_positive]
        min_z = np.min(tree_coords[:, 2])
        z_thresh = min_z + 0.30
        lowest_points = tree_coords[tree_coords[:, 2] <= z_thresh]
        position = np.mean(lowest_points, axis=0)
        position = position[:2]
        coords_centered = coords[:, :2] - position

        # relative distance to seedpoint (0=seedpoint, 1=most distant point)
        distance_from_seedpoint = np.linalg.norm(coords_centered, ord=None, axis=1)
        distance_from_seedpoint_tree = distance_from_seedpoint[ind_positive]
        sorted_inds = distance_from_seedpoint_tree.argsort()
        regularized_max = distance_from_seedpoint_tree[sorted_inds[-5]]
        distance_from_seedpoint = distance_from_seedpoint / regularized_max

        # get precision, recall and F1-score radial
        for i in range(len(intvls) - 1):
            ind_gte_min = distance_from_seedpoint >= intvls[i]
            ind_lt_max = distance_from_seedpoint < intvls[i+1]
            ind_orbit = ind_gte_min & ind_lt_max
            ind_positive_orbit = ind_positive[ind_orbit]
            ind_pred_positive_orbit = ind_pred_positive[ind_orbit]
            tp, fp, tn, fn = get_eval_components(ind_pred_positive_orbit, ind_positive_orbit)
            prec, rec, iou = get_segmentation_metrics(tp, fp, fn)

            # append scores to validation results
            val_res[f'prec_intvl{intvls[i]}_{intvls[i+1]}'].append(prec)
            val_res[f'rec_intvl{intvls[i]}_{intvls[i+1]}'].append(rec)
            val_res[f'iou_intvl{intvls[i]}_{intvls[i+1]}'].append(iou)
        
    val_res_df = pd.DataFrame.from_dict(val_res)
    return val_res_df


################################## Instance segmentation evaluation for the z partition of trees
def evaluate_z_partition(instance_preds, instance_labels, unique_gts, unique_preds, coords, intvls, mapping_to_original_gt_nums, mapping_to_original_pred_nums):
    val_res = dict()
    val_res['instance_pred'] = []
    val_res['instance_label'] = []
    for i in range(len(intvls) - 1):
        val_res[f'prec_intvl{intvls[i]}_{intvls[i+1]}'] = []
    for i in range(len(intvls) - 1):
        val_res[f'rec_intvl{intvls[i]}_{intvls[i+1]}'] = []
    for i in range(len(intvls) - 1):
        val_res[f'iou_intvl{intvls[i]}_{intvls[i+1]}'] = []

    # get results
    for instance_pred, instance_label in zip(unique_preds, unique_gts):
        val_res['instance_pred'].append(mapping_to_original_pred_nums[instance_pred])
        val_res['instance_label'].append(mapping_to_original_gt_nums[instance_label])

        ind_pred_positive = instance_preds == instance_pred
        ind_positive = instance_labels == instance_label
        tree_coords = coords[ind_positive]

        # get relative distance to lowest point (0=lowest point, 1=highest point)
        coords_temp = coords - np.array([0, 0, np.min(tree_coords[:, -1])])
        coords_temp_z = coords_temp[:, -1]

        sorted_inds = tree_coords[:, 2].argsort()
        regularized_max = tree_coords[:, 2][sorted_inds[-5]]
        coords_temp_z = coords_temp_z / (regularized_max - np.min(tree_coords[:, -1]))

        # get precision, recall and F1-score vertical
        for i in range(len(intvls) - 1):
            ind_gte_min = coords_temp_z >= intvls[i]
            ind_lt_max = coords_temp_z < intvls[i+1]
            ind_layer = ind_gte_min & ind_lt_max
            ind_positive_layer = ind_positive[ind_layer]
            ind_pred_positive_layer = ind_pred_positive[ind_layer]
            tp, fp, tn, fn = get_eval_components(ind_pred_positive_layer, ind_positive_layer)
            prec, rec, iou = get_segmentation_metrics(tp, fp, fn)

            # append scores to validation results
            val_res[f'prec_intvl{intvls[i]}_{intvls[i+1]}'].append(prec)
            val_res[f'rec_intvl{intvls[i]}_{intvls[i+1]}'].append(rec)
            val_res[f'iou_intvl{intvls[i]}_{intvls[i+1]}'].append(iou)

    val_res_df = pd.DataFrame.from_dict(val_res)
    return val_res_df


################################## Calculate components of evaluation metrics
def get_eval_components(preds_mask, labels_mask):
    assert len(preds_mask) == len(labels_mask)

    tp = (preds_mask & labels_mask).sum()
    fp = (preds_mask & np.logical_not(labels_mask)).sum()
    fn = (np.logical_not(preds_mask) & labels_mask).sum()
    tn = (np.logical_not(preds_mask) & np.logical_not(labels_mask)).sum()

    return tp, fp, tn, fn


################################## Calculate segmentation metrics
def get_segmentation_metrics(tp, fp, fn):
    assert not (np.isnan(tp) or np.isnan(fp) or np.isnan(fn)), 'one of the inputs is nan'
    # iou
    if tp == 0 and fp == 0 and fn == 0:
        iou = np.nan
    else:
        iou = tp / (tp + fp + fn)
    # rec
    if tp + fn == 0:
        rec = np.nan
    else:
        rec = tp / (tp + fn)
    # prec
    if tp + fp == 0:
        prec = np.nan
    else:
        prec = tp / (tp + fp)
    
    return prec, rec, iou
