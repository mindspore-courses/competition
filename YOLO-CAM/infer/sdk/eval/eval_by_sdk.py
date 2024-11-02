import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def get_eval_result(ann_file, result_file):
    """Get eval result."""
    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(result_file)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="yolov5 eval")
    parser.add_argument('--ann_file', type=str, default='', help='path to annotation')
    parser.add_argument('--result_file', type=str, default='', help='path to annotation')

    args = parser.parse_args()

    get_eval_result(args.ann_file, args.result_file)
