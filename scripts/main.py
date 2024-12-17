# Append path
import sys
sys.path.append("../is_gpt_bayesian/")


# Set up logging
from is_gpt_bayesian.utils import time_utils, path_utils
import logging.config

run_name = 'eg'
run_path = path_utils.run_path(run_name)
path_utils.create_path(run_path)
log_path = path_utils.log_path(run_name)

logging.config.fileConfig('logging.conf', defaults={'logfilename': log_path})
logger = logging.getLogger(__name__)

# Import modules
from is_gpt_bayesian.model import OpenAISession
from is_gpt_bayesian.processing import (specs_processing, 
                                        prompt_processing, 
                                        response_processing)
