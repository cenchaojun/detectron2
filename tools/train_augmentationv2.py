#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
# python tools/train.py --config-file configs/Base-RetinaNet.yaml --num-gpus 1  OUTPUT_DIR training_dir/Base-RetinaNet
# python tools/train.py --config-file configs/Base-RetinaNet.yaml --eval-only MODEL.WEIGHTS //data/cenzhaojun/detectron2/training_dir/Base-RetinaNet/model_0014999.pth OUTPUT_DIR training_dir/Base-RetinaNet
# python tools/train.py --config-file configs/Misc/cascade_mask_rcnn_R_50_FPN_1x.yaml --num-gpus 2  OUTPUT_DIR training_dir/cascade_mask_rcnn_R_50_FPN_1x
# python tools/train_debug.py --config-file configs/Base-RetinaNet.yaml --num-gpus 2  OUTPUT_DIR training_dir/Base-RetinaNet
import logging
import os
from collections import OrderedDict
import torch

from skimage.util import random_noise
import random

from detectron2.data import transforms as T
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data import MetadataCatalog,build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
print("hello")
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from detectron2.modeling import GeneralizedRCNNWithTTA
##===============注册自定义数据集================##
from detectron2.data.datasets import register_coco_instances
register_coco_instances("SSLAD-2D_train", {}, json_file=r"/data/cenzhaojun/dataset/SODA10M/SSLAD-2D/labeled/annotations/instance_train.json",
 image_root = r"/data/cenzhaojun/dataset/SODA10M/SSLAD-2D/labeled/train")
register_coco_instances("SSLAD-2D_test", {}, r"/data/cenzhaojun/dataset/SODA10M/SSLAD-2D/labeled/annotations/instance_val.json",
 r"/data/cenzhaojun/dataset/SODA10M/SSLAD-2D/labeled/val")

# 设置类别
from detectron2.data import MetadataCatalog
MetadataCatalog.get("SSLAD-2D_train").thing_classes = ['Pedestrian','Cyclist','Car','Truck','Tram','Tricycle']
MetadataCatalog.get("SSLAD-2D_test").thing_classes = ['Pedestrian','Cyclist','Car','Truck','Tram','Tricycle']


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)

def random_range_noise(img, mode='pepper', range=(0.0, 0.1)):
    amount = random.uniform(range[0], range[1])
    if mode == 'pepper':
        return random_noise(img, mode=mode, amount=amount, clip=True)
    elif mode == 'gaussian':
        return random_noise(img, mode=mode, clip=True, var=amount)

class RandomPepperNoise(T.Augmentation):
    def get_transform(self, image):
        return T.ColorTransform(lambda x: (random_range_noise(x, mode='pepper', range=(0.0, 0.1))*255).astype('uint8'))
class RandomGaussianNoise(T.Augmentation):
    def get_transform(self, image):
        return T.ColorTransform(lambda x: (random_range_noise(x, mode='gaussian', range=(0.0005, 0.01))*255).astype('uint8'))


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=[
                T.RandomBrightness(0.3, 2.0),
                T.RandomContrast(0.3, 2.5),
                # T.ColorTransform
                # RandomGaussianNoise(),
                # RandomPepperNoise(),
                # T.RandomRotation([-90,90]),
                # RandomResize(0.5,1.5),
                # T.RandomCrop('relative_range',(0.3,0.3)),
                # T.RandomExtent(scale_range=(0.3, 1), shift_range=(1, 1))
        ]))
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    args.config_file = '../configs/Base-RetinaNet.yaml'
    cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    # cfg.num_gpus =2

    cfg.DATASETS.TRAIN = ("SSLAD-2D_train",)  # 训练数据集名称
    cfg.DATASETS.TEST = ("SSLAD-2D_test",)

    cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-101.pkl"
    cfg.DATASETS.TRAIN = ("SSLAD-2D_train",)  # 训练数据集名称
    cfg.DATASETS.TEST = ("SSLAD-2D_test",)
    # cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    # python tools/train_debug.py --config-file configs/Base-RetinaNet.yaml --num-gpus 2 OUTPUT_DIR training_dir/Base-RetinaNet
    # parser = default_argument_parser()
    # # 下面这个就可以调用2块GPU的命令行
    # parser.add_argument("--num-gpus", type=int, default=2, help="number of gpus *per machine*")
    # args = parser.parse_args()
    #
    # args = default_argument_parser().parse_args()
    # print("Command Line Args:", args)
    # launch(
    #     main,
    #     args.num_gpus,
    #     num_machines=args.num_machines,
    #     machine_rank=args.machine_rank,
    #     dist_url=args.dist_url,
    #     args=(args,),
    # )
