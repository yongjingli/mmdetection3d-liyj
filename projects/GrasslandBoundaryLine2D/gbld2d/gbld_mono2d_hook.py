from mmengine.hooks import Hook
from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class Gbld2DTransformHook(Hook):
    def __init__(self, name, set_first_epoch):
        self.epoch = 0
        self.max_epoch = None
        self.name = name
        self.set_first_epoch = set_first_epoch

    # def before_run(self, runner) -> None:
    #
    # def after_run(self, runner) -> None:
    #
    # def before_train(self, runner) -> None:
    #
    # def after_train(self, runner) -> None:
    #
    def before_train_epoch(self, runner) -> None:
        if self.set_first_epoch:
            if self.max_epoch is None:
                self.max_epoch = runner._train_loop.max_epochs

            for transform in runner.train_dataloader.dataset.pipeline.transforms:
                if hasattr(transform, 'name'):
                    if transform.name in ["GgldRandomCrop"]:
                        # 设置数据增强的概率从[0, transform.prob]
                        transform.prob = transform.st_prob * self.epoch/self.max_epoch
                        # print("set ", transform.name, 'prob:', transform.prob)
            self.set_first_epoch = False

    def after_train_epoch(self, runner) -> None:
        self.epoch = self.epoch + 1

        if self.max_epoch is None:
            self.max_epoch = runner._train_loop.max_epochs

        for transform in runner.train_dataloader.dataset.pipeline.transforms:
            if hasattr(transform, 'name'):
                if transform.name in ["GgldRandomCrop"]:
                    # 设置数据增强的概率从[0, transform.prob]
                    transform.prob = transform.st_prob * self.epoch / self.max_epoch
                    # print("set ", transform.name, 'prob:', transform.prob)

    # def before_train_iter(self,
    #                       runner,
    #                       batch_idx: int,
    #                       data_batch: DATA_BATCH = None) -> None:
    #
    # def after_train_iter(self,
    #                      runner,
    #                      batch_idx: int,
    #                      data_batch: DATA_BATCH = None,
    #                      outputs: Optional[dict] = None) -> None: