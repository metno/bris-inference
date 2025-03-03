from bris.utils import get_base_seed
import os

def test_get_base_seed():
    # Set up test vars
    os.environ["AIFS_BASE_SEED"]="1234"

    seed = get_base_seed(env_var_list=("AIFS_BASE_SEED", "SLURM_JOB_ID"))

    assert isinstance(seed, int)
    assert seed > 1000


if __name__ == "__main__":
    test_get_base_seed()
