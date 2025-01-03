import os
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


def run_results_file_path(run_name, return_posix=True):
    _check_bool(return_posix)
    if return_posix:
        return run_results_file_path(run_name, return_posix=False).as_posix()
    else:
        return run_path(run_name, return_posix=False) / "run_results_file.csv"


def run_final_stacked_file_path(run_name, return_posix=True):
    _check_bool(return_posix)
    if return_posix:
        return run_final_stacked_file_path(run_name, return_posix=False).as_posix()
    else:
        return run_path(run_name, return_posix=False) / "run_final_stacked_file.csv"


def run_final_stacked_processed_file_path(run_name, return_posix=True):
    _check_bool(return_posix)
    if return_posix:
        return run_final_stacked_processed_file_path(run_name, return_posix=False).as_posix()
    else:
        return run_path(run_name, return_posix=False) / "run_final_stacked_processed_file.csv"


def run_final_unstacked_file_path(run_name, return_posix=True):
    _check_bool(return_posix)
    if return_posix:
        return run_final_unstacked_file_path(run_name, return_posix=False).as_posix()
    else:
        return run_path(run_name, return_posix=False) / "run_final_unstacked_file.csv"


def run_final_unstacked_ungrouped_file_path(run_name, group_name, return_posix=True):
    _check_bool(return_posix)
    if return_posix:
        return run_final_unstacked_ungrouped_file_path(run_name, group_name, return_posix=False).as_posix()
    else:
        return run_path(run_name, return_posix=False) / f"run_final_unstacked_{group_name}_file.csv"


def run_final_unstacked_ungrouped_mat_file_path(run_name, group_name, return_posix=True):
    _check_bool(return_posix)
    if return_posix:
        return run_final_unstacked_ungrouped_mat_file_path(run_name, group_name, return_posix=False).as_posix()
    else:
        return run_path(run_name, return_posix=False) / f"run_final_unstacked_{group_name}_mat_file.csv"


def run_final_unstacked_ungrouped_subject_file_path(run_name, group_name, return_posix=True):
    _check_bool(return_posix)
    if return_posix:
        return run_final_unstacked_ungrouped_subject_file_path(run_name, group_name, return_posix=False).as_posix()
    else:
        return run_path(run_name, return_posix=False) / f"run_final_unstacked_{group_name}_subject_file.csv"


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


def job_results_file_path(job_path, return_posix=True):
    _check_bool(return_posix)
    if return_posix:
        return job_results_file_path(job_path, return_posix=False).as_posix()
    else:
        return _convert_to_path(job_path) / "job_results_file.csv"


def create_path(path, exist_ok=True) -> None:
    path = _convert_to_path(path)
    path.mkdir(parents=True, exist_ok=exist_ok)


def get_subdirs(path, return_posix=True) -> list:
    _check_bool(return_posix)
    if return_posix:
        result = [d.as_posix() for d in _convert_to_path(path).iterdir() if d.is_dir()]
        result.sort()
        return result
    else:
        result = [d for d in _convert_to_path(path).iterdir() if d.is_dir()]
        result.sort()
        return result
    

def rename_with_index(file_path):
    
    if not os.path.exists(file_path):
        return
    
    dir_name, filename = os.path.split(file_path)
    base, ext = os.path.splitext(filename)
    
    index = 0
    while True:
        new_name = f"{base}_{index}{ext}"
        new_path = os.path.join(dir_name, new_name)
        
        if not os.path.exists(new_path):
            os.rename(file_path, new_path)
            break
        
        index += 1


def _check_bool(return_posix):
    if not isinstance(return_posix, bool):
        raise ValueError('return_posix must be a boolean.')
    
    
def _convert_to_path(path):
    if isinstance(path, Path):
        return path
    else:
        return Path(path)
    
