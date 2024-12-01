import logging
import yaml
from abc import abstractmethod
from yacs.config import CfgNode as CN


from eo_lib.utils.cuda import get_device
from eo_lib.utils.reproducibility import set_random_seed
from eo_lib.pipeline.batch_processing_pipe import PretrainingDataPipe
from eo_lib.datasets.multimodal_ds.preprocess import read_split_file
from eo_lib.utils.read_config import load_cfg

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class Pipeline:
    def __init__(self, data_cfg_file: str, model_cfg_file: str | None = None) -> None:

        self._logger = logging.getLogger(Pipeline.__qualname__)

        device = get_device()
        self._logger.info(f"Device that will be used is : {device}")
        if model_cfg_file:
            self.model_cfg = load_cfg(model_cfg_file)
            self._logger.info("Model Config Loaded")
        self.data_config = load_cfg(data_cfg_file)
        self._logger.info("Data Config Loaded")
        set_random_seed(self.data_config.random_seed)

    def _build_datamodule(self):
        training_files = read_split_file(self.data_config.file_path)
        if self.data_config.train_step == "hpo":
            test_files = read_split_file(self.data_config.val_path)
        else:
            test_files = None
        self.data_module_ins = PretrainingDataPipe(
            files=training_files,
            test_files=test_files,
            num_workers=self.data_config.num_workers,
            add_context=self.data_config.add_context,
            pin_memory=self.data_config.pin_memory,
            prefetch_factor=self.data_config.prefetch_factor,
            step=self.data_config.train_step,
            batch_size=self.data_config.batch_size,
        )
        self.data_module_ins.setup()

    @abstractmethod
    def _build_model(self):

        raise NotImplementedError

    @abstractmethod
    def train(self):
        self._logger.info("Starting training")

        if not hasattr(self, "trainer"):
            raise AttributeError("Trainer should be defined in _build_model")

        self.trainer.fit(
            self.model,
            train_dataloaders=self.data_module_ins.train_dataloader(),
            val_dataloaders=self.data_module_ins.val_dataloader()
            if self.model_cfg.step == "hpo"
            else None,
            ckpt_path=(
                self.model_cfg.trainer.checkpoint_path
                if self.model_cfg.trainer.load_from_checkpoint
                else "last"
            ),
        )

        self._logger.info("Logged Metrics:", self.trainer.logged_metrics)
        self._logger.info("Callback Metrics:", self.trainer.callback_metrics)

        self.trainer.logger.finalize("done")

        self._logger.info("Train process complete")

    def setup(self):
        self._build_datamodule()
        self._logger.info("Data Module Initialized")

        self.model = self._build_model()
        self._logger.info(f"Objective Loaded: {self.model}")
