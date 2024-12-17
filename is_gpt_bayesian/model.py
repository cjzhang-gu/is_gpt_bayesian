import openai
import logging
import pickle
import json
from pathlib import Path
from is_gpt_bayesian.utils import time_utils, path_utils

logger = logging.getLogger(__name__)


class OpenAISession():

    def __init__(self, run_name):
        self.client = openai.OpenAI()
        self.run_name = run_name
        self.jobs = {}
        logger.info(f'OpenAI session created with run_name {self.run_name}.')
