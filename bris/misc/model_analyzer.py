import torch
import pytorch_lightning as pl

from omegaconf import OmegaConf

from bris.misc.modelblocks import ModelBlocks
from bris.checkpoint import Checkpoint
from bris.data.datamodule import DataModule
from bris.utils import create_config
from bris.ddp_strategy import DDPGroupStrategy


MYCONFIG = "/pfs/lustrep4/scratch/project_465000527/salihiar/bris-inference/stretched-grid.yaml"  #'working_example.yaml'


# Loading config and initialiazing the checkpoint class for metadata

try:
    config = OmegaConf.load(MYCONFIG)
except Exception as e:
    raise e
checkpoint = Checkpoint(config.checkpoint_path)

datamodule = DataModule(
    config=config,
    checkpoint_object=checkpoint,
)

trainer = pl.Trainer(
    logger=False,
    limit_predict_batches=2,  # controls num predictions
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    deterministic=False,
    detect_anomaly=False,
    strategy=DDPGroupStrategy(
        num_gpus_per_model=config.hardware.num_gpus_per_model,
        read_group_size=1,
        static_graph=False,
    ),
    devices=config.hardware.num_gpus_per_node,
    num_nodes=config.hardware.num_nodes,
    precision="bf16",
    inference_mode=True,
    use_distributed_sampler=False,
    callbacks=None,
)

run_model_blocks = ModelBlocks(checkpoint=checkpoint, which_block="processor")


output = trainer.predict(
    run_model_blocks, datamodule=datamodule, return_predictions=True
)
