import os
import torch
from typing import Dict, List

from utils.utils_det import configure_logger
from detectron2.data.catalog import MetadataCatalog
import hydra
from detectron2 import model_zoo
from omegaconf import OmegaConf, DictConfig
import detectron2.utils.comm as comm

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from collections import OrderedDict

from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.data import (DatasetCatalog, DatasetMapper,
                             build_detection_train_loader,
                             build_detection_test_loader)

from detectron2.engine import DefaultTrainer, launch, default_setup, DefaultPredictor,default_argument_parser,hooks
from detectron2.data import transforms as T

# from data_loading import conflab_dataset
from utils import utils_dist, utils_slurm, create_train_augmentation, create_test_augmentation
import rich
import logging
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



logger = logging.getLogger("detectron2")
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


class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg,
                               is_train=True,
                               augmentations=create_train_augmentation(cfg))
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = DatasetMapper(cfg,
                               is_train=False,
                               augmentations=create_test_augmentation(cfg))
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name,
                             cfg.TASKS,
                             False,
                             output_dir=output_folder)
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


class Predictor(DefaultPredictor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.aug = T.Resize((cfg.image_h, cfg.image_w))


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    args.config_file = '../configs/Base-RetinaNet.yaml'
    cfg.merge_from_file(args.config_file)

    cfg.DATASETS.TRAIN = ("SSLAD-2D_train",)  # 训练数据集名称
    cfg.DATASETS.TEST = ("SSLAD-2D_test",)

    cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-101.pkl"
    cfg.DATASETS.TRAIN = ("SSLAD-2D_train",)  # 训练数据集名称
    cfg.DATASETS.TEST = ("SSLAD-2D_test",)

    #cfg.merge_from_file(model_zoo.get_config_file(args.model_zoo))
    #cfg.DATASETS.TRAIN = (args.train_dataset, )
    #cfg.DATASETS.TEST = (args.test_dataset, )
    cfg.DATALOADER.NUM_WORKERS = 6

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
    cfg.OUTPUT_DIR = '/data/cenzhaojun/detectron2/training_dir/Base-RetinaNet'
    # cfg.image_w = args.size[0]
    # cfg.image_h = args.size[1]
    cfg.image_w_test = 1200
    cfg.image_h_test = 1200
    cfg.half_crop = 'false'

    # cfg.TASKS = tuple(args.eval_task)

    cfg.SOLVER.REFERENCE_WORLD_SIZE = 1
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000

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

# def main(args: DictConfig):
#     args = OmegaConf.create(OmegaConf.to_yaml(args, resolve=True))
#
#     rich.print("Command Line Args:\n{}".format(
#         OmegaConf.to_yaml(args, resolve=True)))
#
#     if args.accelerator == "ddp":
#         utils_dist.init_distributed_mode(args)
#
#     # register dataset
#     # conflab_dataset.register_conflab_dataset(args)
#
#     if args.create_coco:
#         pass
#         # only create dataset
#         # return
#
#     cfg = setup(args)
#
#     if args.eval_only is False:
#         configure_logger(args, fileonly=True)
#
#         trainer = Trainer(cfg)
#         trainer.resume_or_load(resume=args.resume)
#         trainer.train()
#
#     else:
#         # setup logger
#         configure_logger(args)
#
#         if args.visualize is False:
#             model = Trainer.build_model(cfg)
#             DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
#             res = Trainer.test(cfg, model)
#             logger.info(res)
#             return res
#         else:
#             test_dataset: List[Dict] = DatasetCatalog.get(args.test_dataset)
#             metadata = MetadataCatalog.get(args.test_dataset)
#             predictor = Predictor(cfg)
#             visualize_det2(test_dataset,
#                            predictor,
#                            metadata=metadata,
#                            vis_conf=args.vis)



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
print("helll")
# def main_spawn(args: DictConfig):
#     # ddp spawn
#     launch(main,
#            args.ngpus,
#            machine_rank=args.machine_rank,
#            num_machines=args.num_machines,
#            dist_url=args.dist_url,
#            args=(args, ))
#
#
# @hydra.main(config_name='config', config_path='conf')
# def hydra_main(args: DictConfig):
#     if args.launcher_name == "local":
#         if args.accelerator == "ddp":
#             main(args)
#         else:
#             args.dist_url = "auto"
#             main_spawn(args)
#     elif args.launcher_name == "slurm":
#         from utils.utils_slurm import submitit_main
#         submitit_main(args)
#
#
# if __name__ == "__main__":
#     hydra_main()
