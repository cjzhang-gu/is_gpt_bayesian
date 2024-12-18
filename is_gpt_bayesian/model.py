import os
import openai
import logging
import pickle
import json
from pathlib import Path
import pandas as pd
from is_gpt_bayesian.utils import time_utils, path_utils

logger = logging.getLogger(__name__)


class OpenAISession():

    def __init__(self, run_name):
        self.client = openai.OpenAI()
        self.run_name = run_name
        self.jobs = {}
        logger.info(f'OpenAI session created with run_name {self.run_name}.')


    def generate_batch_files(self, specs_df) -> dict:

        run_specs_filename = path_utils.run_specs_file_path(self.run_name)
        specs_df.to_csv(run_specs_filename)

        # split by models
        for model_name, model_specs_df in specs_df.groupby('model'):
            
            # check length
            if len(model_specs_df) > 50_000:
                raise ValueError('Cannot process batches > 50,000 queries.')

            # create jsonl file
            jsonl_list = []
            for idx, one_spec in model_specs_df.iterrows():
                request = {"custom_id": f"request-{idx+1}",
                           "method"   : "POST",
                           "url"      : "/v1/chat/completions",
                           "body"     : {"model"   : model_name,
                                         "messages": [{"role": "user", "content": one_spec['prompt']}],
                                         }
                           }
                if 'temperature' in one_spec:
                    request['body']['temperature'] = one_spec['temperature']
                if 'seed' in one_spec:
                    request['body']['seed'] = one_spec['seed']

                jsonl_list.append(request)
            
            # create job path
            job_path = path_utils.job_path(run_name=self.run_name, job_name=model_name)
            path_utils.create_path(job_path, exist_ok=False)
            logger.info(f"Directory created {job_path}.")

            # save specs_df 
            job_specs_filename  = path_utils.job_specs_file_path(job_path)
            model_specs_df.to_csv(job_specs_filename)
            logging.info(f"Specs dataframe saved to {job_specs_filename}.")

            # save source file
            job_source_filename = path_utils.job_source_file_path(job_path)
            with open(job_source_filename, "w") as file:
                for obj in jsonl_list:
                    file.write(json.dumps(obj) + '\n')
            logging.info(f"Source file saved to {job_source_filename}")
            
            self.jobs[model_name] = job_path

        return self.jobs
        

    def send_batches(self):
        for _, job_path in self.jobs.items():
            self.send_one_batch(self, job_path)


    def send_one_batch(self, job_path):

        source_filename = path_utils.job_source_file_path(job_path)

        # create file object
        batch_input_file = self.client.files.create(file=open(source_filename, "rb"),
                                                    purpose="batch"
                                                    )

        # send batch job
        batch_obj = self.client.batches.create(input_file_id=batch_input_file.id,
                                                endpoint="/v1/chat/completions",
                                                completion_window="24h",
                                                metadata={"description": "eval job"}
                                                )
        logger.info(f"Batch job {batch_obj.id} sent to OpenAI.")
        
        # save info file
        job_info_filename = path_utils.job_info_file_path(job_path)
        with open(job_info_filename, "w") as file:
            json.dump(_obj_to_json_dict_helper(batch_obj), file, indent=4)
        logging.info(f"Info file saved to {job_info_filename}.")



    def load_jobs(self):
        job_paths = path_utils.get_subdirs(path_utils.run_path(self.run_name), return_posix=False)
        self.jobs = {d.name.split("__")[1]: d.as_posix() for d in job_paths}
        logger.info(f'Session {self.run_name} loaded previous jobs info:')
        logger.info(f'{self.jobs}')
        return self.jobs
    

    def retrieve_batches(self):
        summary = {}
        for _, job_path in self.jobs.items():
            summary.update(self.retrieve_one_batch(job_path))


    def retrieve_one_batch(self, job_path):

        # read info file
        info_filename = path_utils.job_info_file_path(job_path)
        with open(info_filename, "r") as file:
            job_info = json.load(file)

        # check job status
        # - job was in progress
        if job_info['stutus'] in ['validating', 'in_progress', 'finalizing']:
            
            # check job
            batch_obj = self.client.batches.retrieve(job_info['id'])
            job_info_filename = path_utils.job_info_file_path(job_path)
            with open(job_info_filename, "w") as file:
                json.dump(_obj_to_json_dict_helper(batch_obj), file, indent=4)
            logging.info(f"Info file saved to {job_info_filename}.") 

            # job still in progress
            if batch_obj.status in ['validating', 'in_progress', 'finalizing']:
                logger.info(f"Run: {self.run_name}, job: {job_path} is {batch_obj.status}.")
            
            # job completed
            elif batch_obj.status == 'completed':
                job_response = self.client.files.content(batch_obj.output_file_id).content.decode('utf-8')
                job_response_filename = path_utils.job_response_file_path(job_path)
                with open(job_response_filename, "w") as file:
                    file.write(str(job_response))
                logger.info(f"Run: {self.run_name}, job: {job_path} is now completed and response is saved to {job_response_filename}.")

            # job unknown status, treat as error
            else:
                logger.error(f"[UNKNOWN STATUS] Run: {self.run_name}, job: {job_path} is {batch_obj.status}.")

            return {job_path: batch_obj.status}

        # - job was completed, nothing to do
        elif job_info['status'] == 'completed':
            logger.log(f'Run: {self.run_name}, job: {job_path} was previous completed.')

            return {job_path: 'completed'}

        # - unknown job status, treat as error
        else:
            logger.error(f"[UNKNOWN STATUS] Run: {self.run_name}, job: {job_path} was previouly {job_info['status']}.")

            return {job_path: job_info['status']}


    def resend_failed_jobs(self):

        for _, job_path in self.jobs.items():

            # read info file
            info_filename = path_utils.job_info_file_path(job_path)
            with open(info_filename, "r") as file:
                job_info = json.load(file)

            # if in error, resend batch
            if job_info['status'] not in ['completed', 'validating', 'in_progress', 'finalizing']:
                logger.info(f'Re-sending {job_info['status']} job {job_path} ...')
                self.send_one_batch(job_path)


    def process_reponses(self):
        raise NotImplementedError


    def process_one_response(self, job_path):
        raise NotImplementedError

        # # read response file
        # job_response_filename = path_utils.job_response_file_path(job_path)
        # with open(job_response_filename, 'r') as file:
        #     job_response_content = file.read()
        # job_response_content = [json.loads(r) for r in job_response_content.split('\n') if r != ""]
        
        # # read specs df
        # job_specs_filename = path_utils.job_specs_file_path(job_path)
        # job_specs_df = pd.read_csv(job_specs_filename, index_col=0)

        # if any(col in job_specs_df.columns for col in ['batch_id', 'request_id', 'textual_response']):
        #     raise NameError('Some or all')
        
        # job_specs_df['batch_id'] = [r['id'] for r in job_response_content]
        # job_specs_df['request_id'] = [r['response']['request_id'] for r in job_response_content]
        # job_specs_df['textual_response'] = [r['response']['body']['choices'][-1]['message']['content'] for r in job_response_content]
        
        # job_results_filename = path_utils.job_results_file_path(job_path)
        # job_specs_df.to_csv(job_results_filename)
        # logger.info(f'Results file saved to {job_results_filename}')


def _obj_to_json_dict_helper(obj):
    def is_json_serializable(value):
        try:
            json.dumps(value)
            return True
        except (TypeError, OverflowError):
            return False
    return {k: v if is_json_serializable(v) else str(v) for k,v in vars(obj).items()}
    

def make_file_read_only(file_path):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    os.chmod(path, 0o444)