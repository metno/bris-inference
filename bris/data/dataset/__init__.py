import logging

from torch.utils.data import get_worker_info

# Make NativeGridDataset and ZipDataset importable from bris.data.dataset:
from .nativegrid import NativeGridDataset  # noqa
from .zip import ZipDataset  # noqa

LOGGER = logging.getLogger(__name__)


def worker_init_func(worker_id: int) -> None:
    """Configures each dataset worker process. This is a helper function, called by Dataloader.

    Calls WeatherBenchDataset.per_worker_init() on each dataset object.

    Parameters
    ----------
    worker_id : int
        Worker ID

    Raises
    ------
    RuntimeError
        If worker_info is None
    """
    worker_info = get_worker_info()  # information specific to each worker process
    if worker_info is None:
        LOGGER.error("worker_info is None! Set num_workers > 0 in your dataloader!")
        raise RuntimeError
    dataset_obj = (
        worker_info.dataset
    )  # the copy of the dataset held by this worker process.
    dataset_obj.per_worker_init(
        n_workers=worker_info.num_workers,
        worker_id=worker_id,
    )
