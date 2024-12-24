import os
import openai
import logging
import pickle
import pprint
import json
from pathlib import Path
import pandas as pd
from is_gpt_bayesian.utils import time_utils, path_utils

logger = logging.getLogger(__name__)


class OpenAISession():

    def __init__(self, run_name):
        self.client = openai.OpenAI()
        self.run_name = run_name
        self.jobs = []
        logger.info(f'OpenAI session created with run_name {self.run_name}.')


    def generate_batch_files(self, specs_df) -> dict:
        
        run_specs_filename = path_utils.run_specs_file_path(self.run_name)
        path_utils.rename_with_index(run_specs_filename)
        specs_df.to_csv(run_specs_filename)
        make_file_read_only(run_specs_filename)
        logger.info(f'Run specs has been specified:\n {specs_df}')
        logger.info(f'Run specs has been saved to: {run_specs_filename}.')

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
            make_file_read_only(job_specs_filename)
            logger.info(f"Specs dataframe saved to {job_specs_filename}.")

            # save source file
            job_source_filename = path_utils.job_source_file_path(job_path)
            with open(job_source_filename, "w") as file:
                for obj in jsonl_list:
                    file.write(json.dumps(obj) + '\n')
            make_file_read_only(job_source_filename)
            logger.info(f"Source file saved to {job_source_filename}")
            
            self.jobs.append(job_path)

        return self.jobs
        

    def send_batches(self):
        summary = {}

        for job_path in self.jobs:
            summary.update(self.send_one_batch(job_path))
        logger.info(f'---------- JOBS SENT SUMMARY\n{pprint.pformat(summary)}')


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
        logger.info(f"Info file saved to {job_info_filename}.")

        return {job_path: 'sent'}


    def load_jobs(self):
        self.jobs = path_utils.get_subdirs(path_utils.run_path(self.run_name), return_posix=True)
        logger.info(f'Session {self.run_name} loaded previous jobs info:')
        logger.info(f'---------- JOBS LOADED SUMMARY\n{pprint.pformat(self.jobs)}')
        return self.jobs
    

    def retrieve_batches(self):
        summary = {}
        for job_path in self.jobs:
            summary.update(self.retrieve_one_batch(job_path))
        logger.info(f'---------- JOBS RETRIEVED SUMMARY\n{pprint.pformat(summary)}')


    def retrieve_one_batch(self, job_path):

        # read info file
        job_info_filename = path_utils.job_info_file_path(job_path)
        with open(job_info_filename, "r") as file:
            job_info = json.load(file)

        # check job status
        # - job was in progress
        if job_info['status'] in ['validating', 'in_progress', 'finalizing']:
            
            # check job
            batch_obj = self.client.batches.retrieve(job_info['id'])
            job_info_filename = path_utils.job_info_file_path(job_path)
            with open(job_info_filename, "w") as file:
                json.dump(_obj_to_json_dict_helper(batch_obj), file, indent=4)
            logger.info(f"Info file saved to {job_info_filename}.") 

            # job still in progress
            if batch_obj.status in ['validating', 'in_progress', 'finalizing']:
                logger.info(f"Run: {self.run_name}, job: {job_path} is {batch_obj.status}.")
            
            # job completed
            elif batch_obj.status == 'completed':
                job_response = self.client.files.content(batch_obj.output_file_id).content.decode('utf-8')
                job_response_filename = path_utils.job_response_file_path(job_path)
                with open(job_response_filename, "w") as file:
                    file.write(str(job_response))
                make_file_read_only(job_response_filename)
                logger.info(f"Run: {self.run_name}, job: {job_path} is now completed and response is saved to {job_response_filename}.")

            # job unknown status, treat as error
            else:
                logger.warning(f"[UNKNOWN STATUS] Run: {self.run_name}, job: {job_path} is {batch_obj.status}.")

            return {job_path: batch_obj.status}

        # - job was completed, nothing to do
        elif job_info['status'] == 'completed':
            logger.info(f'Run: {self.run_name}, job: {job_path} was previous completed.')

            return {job_path: 'completed'}

        # - unknown job status, treat as error
        else:
            logger.warning(f"[UNKNOWN STATUS] Run: {self.run_name}, job: {job_path} was previouly {job_info['status']}.")

            return {job_path: job_info['status']}


    def resend_failed_jobs(self):

        if self.all_completed():
            logger.info('ALL JOBS ARE COMPLETED. NO FAILED JOBS.')

        for job_path in self.jobs:

            # read info file
            info_filename = path_utils.job_info_file_path(job_path)
            with open(info_filename, "r") as file:
                job_info = json.load(file)

            # if in error, resend batch
            if job_info['status'] not in ['completed', 'validating', 'in_progress', 'finalizing']:
                logger.info(f'Re-sending {job_info['status']} job {job_path} ...')
                self.send_one_batch(job_path)


    def process_reponses(self):
        summary = {}
        result_dfs = []

        for job_path in self.jobs:
            job_summary, job_result_df = self.process_one_response(job_path)
            summary.update(job_summary)
            if job_result_df is not None:
                result_dfs.append(job_result_df)
        logger.info(f'---------- RESULTS SUMMARY\n{pprint.pformat(summary)}')

        if result_dfs:
            run_results_filename = path_utils.run_results_file_path(self.run_name)
            run_results_df = pd.concat(result_dfs, axis=0).sort_values(by=['obs_idx', 'created_time'])
            groupby_cols = ['obs_idx', 'trial_id', 'subject_id', 'model', 'instruction'] + \
                run_results_df.columns.intersection(['temperature', 'seed']).to_list()
            run_results_df['with_response'] = run_results_df['textual_response'].notna().astype(int)
            run_results_df['query_idx'] = run_results_df.groupby(groupby_cols)['with_response'].cumsum()
            run_results_df['query_total_count'] = run_results_df.groupby(groupby_cols)['query_idx'].transform('max')
            run_results_df = run_results_df.drop(columns=['with_response'])
            run_results_df.to_csv(run_results_filename)
            logger.info(f'Completed results saved to {run_results_filename}.')

            if self.all_completed():
                logger.info('ALL JOBS ARE COMPLETED. SHOULD CHECK IF RESEND_INVALID IS NECESSARY.')

            return run_results_df
        

    def process_one_response(self, job_path):
        # read response file
        job_specs_filename = path_utils.job_specs_file_path(job_path)
        job_response_filename = path_utils.job_response_file_path(job_path)
        job_results_filename = path_utils.job_results_file_path(job_path)

        # not yet completed
        if not os.path.exists(job_response_filename):
            return {job_path: 'Response file not found.'}, None
        
        # previously completed
        if os.path.exists(job_results_filename):
            job_results_df = pd.read_csv(job_results_filename, index_col=0)
            return {job_path: f'Responses previously proceed. Saved as {job_results_filename}'}, job_results_df

        # just compelted
        with open(job_response_filename, 'r') as file:
            job_response_content = file.read()
        job_response_content = [json.loads(r) for r in job_response_content.split('\n') if r != ""]
        
        # read specs df
        job_results_df = pd.read_csv(job_specs_filename, index_col=0)

        if any(col in job_results_df.columns for col in ['batch_id', 'request_id', 'textual_response']):
            raise RuntimeError('The specs df may have invalid columns, this may happen when resending queries that previously had invalid responses. Please check')

        job_response_content_df = pd.DataFrame({'batch_id': [r['id'] for r in job_response_content],
                                                'request_id': [r['response']['request_id'] for r in job_response_content],
                                                'created_time': [r['response']['body']['created'] for r in job_response_content],
                                                'textual_response': [r['response']['body']['choices'][-1]['message']['content'] for r in job_response_content]},
                                                index=[r['custom_id'] for r in job_response_content],)
        job_response_content_df.index = job_response_content_df.index.str.replace('request-', '').astype(int) - 1

        job_results_df = job_results_df.join(job_response_content_df)
        job_results_df.to_csv(job_results_filename)
        make_file_read_only(job_results_filename)
        logger.info(f'Results file saved to {job_results_filename}')

        return {job_path: f'Results saved to {job_results_filename}.'}, job_results_df
    

    def all_completed(self):
        self.load_jobs()
        all_jobs_status = []
        for job_path in self.jobs:
            # read info file
            job_info_filename = path_utils.job_info_file_path(job_path)
            with open(job_info_filename, "r") as file:
                job_info = json.load(file)
            all_jobs_status.append(job_info['status'] == 'completed')
        return all(all_jobs_status)


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