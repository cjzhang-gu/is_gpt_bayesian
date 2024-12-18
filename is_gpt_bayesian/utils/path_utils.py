from pathlib import Path
from is_gpt_bayesian.utils import time_utils


data_eg_path = Path('assets/datastruct_wisconsin.mat')
data_hs_path = Path('assets/data_holt_and_smith.xlsx')


def run_path(run_name, return_posix=True):
    _check_bool(return_posix)
    if return_posix:
        return run_path(run_name, return_posix=False).as_posix()
    else:
        return Path("runs") / f"{run_name}"


def log_path(run_name, return_posix=True):
    _check_bool(return_posix)
    if return_posix:
        return log_path(run_name, return_posix=False)
    else:
        return run_path(run_name, return_posix=False) / f"{run_name}.log"


def job_path(run_name, job_name, return_posix=True):
    _check_bool(return_posix)
    if return_posix:
        return job_path(run_name, job_name, return_posix=False).as_posix()
    else:
        return run_path(run_name, return_posix=False) / ("job__" + job_name + "__" + time_utils.get_secondstamp())


def run_specs_file_path(run_name, return_posix=True):
    _check_bool(return_posix)
    if return_posix:
        return run_specs_file_path(run_name, return_posix=False).as_posix()
    else:
        return run_path(run_name, return_posix=False) / "run_specs_file.csv"


def job_specs_file_path(job_path, return_posix=True):
    _check_bool(return_posix)
    if return_posix:
        return job_specs_file_path(job_path, return_posix=False).as_posix()
    else:
        return _convert_to_path(job_path) / "job_specs_file.csv"
    

def job_source_file_path(job_path, return_posix=True):
    _check_bool(return_posix)
    if return_posix:
        return job_source_file_path(job_path, return_posix=False).as_posix()
    else:
        return _convert_to_path(job_path) / "job_source_file.jsonl"


def job_info_file_path(job_path, return_posix=True):
    _check_bool(return_posix)
    if return_posix:
        return job_info_file_path(job_path, return_posix=False).as_posix()
    else:
        return _convert_to_path(job_path) / "job_info_file.json"


def job_response_file_path(job_path, return_posix=True):
    _check_bool(return_posix)
    if return_posix:
        return job_response_file_path(job_path, return_posix=False).as_posix()
    else:
        return _convert_to_path(job_path) / "job_response_file.jsonl"


def create_path(path, exist_ok=True) -> None:
    path = _convert_to_path(path)
    path.mkdir(parents=True, exist_ok=exist_ok)


def get_subdirs(path, return_posix=True) -> list:
    _check_bool(return_posix)
    if return_posix:
        return [d.as_posix() for d in _convert_to_path(path).iterdir() if d.is_dir()]
    else:
        return [d for d in _convert_to_path(path).iterdir() if d.is_dir()]


def _check_bool(return_posix):
    if not isinstance(return_posix, bool):
        raise ValueError('return_posix must be a boolean.')
    
    
def _convert_to_path(path):
    if isinstance(path, Path):
        return path
    else:
        return Path(path)
    
