import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

def evaluate_point_cloud(pred_path, gt_path, thresholds=[0.01, 0.02]):
    """
    点云评估函数
    :param pred_path: 预测点云文件路径 (.ply)
    :param gt_path:   真实点云文件路径 (.ply)
    :param thresholds: 评估阈值列表 (米)，如 [0.01, 0.02] = [1cm, 2cm]
    :return: dict 评估结果
    """
    pred_pcd = o3d.io.read_point_cloud(pred_path)
    gt_pcd = o3d.io.read_point_cloud(gt_path)

    pred_points = np.asarray(pred_pcd.points)
    gt_points = np.asarray(gt_pcd.points)
    pred_tree = cKDTree(pred_points)
    gt_tree = cKDTree(gt_points)
    dist_pred_to_gt, _ = gt_tree.query(pred_points)  # accuracy_mean
    dist_gt_to_pred, _ = pred_tree.query(gt_points)  # completeness_mean

    results = {
        "accuracy_mean": float(np.mean(dist_pred_to_gt)),
        "completeness_mean": float(np.mean(dist_gt_to_pred))
    }
    for tau in thresholds:
        precision = np.mean(dist_pred_to_gt < tau)
        recall = np.mean(dist_gt_to_pred < tau)
        if precision + recall > 0:
            fscore = 2 * precision * recall / (precision + recall)
        else:
            fscore = 0.0
        results[f"F-score@{tau}m"] = fscore
        results[f"precision@{tau}m"] = precision
        results[f"recall@{tau}m"] = recall

    return results


if __name__ == "__main__":
    pred_path = "/home/outbreak/mindspore/MVSNet_mindspore/tank_outputs/trainply/Barn.ply"
    gt_path = "/media/outbreak/68E1-B517/Dataset/TankandTemples/offical_training/Barn.ply"
    metrics = evaluate_point_cloud(pred_path, gt_path)
    print("点云评估结果：")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")
