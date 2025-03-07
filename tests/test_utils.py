import os
import time
import bris.utils


def test_expand_time_tokens():
    unixtime = time.time()
    t = bris.utils.expand_time_tokens(filename="/test_file%Y-%m-%d", unixtime=unixtime)
    assert time.strftime("test_file%Y-%m-%d") in t, f"Time tokens not found in {t}."


def test_get_base_seed():
    # Set up test vars
    os.environ["AIFS_BASE_SEED"] = "1234"

    seed = bris.utils.get_base_seed(env_var_list=("AIFS_BASE_SEED", "SLURM_JOB_ID"))

    assert isinstance(seed, int)
    assert seed > 1000


if __name__ == "__main__":
    test_expand_time_tokens()
    test_get_base_seed()
