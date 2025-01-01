if __name__ == '__main__':

    # ===================================
    # Append path
    # ===================================

    import sys
    sys.path.append("../is_gpt_bayesian/")


    # ===================================
    # Resovle args and set up logging
    # ===================================

    import argparse

    parser = argparse.ArgumentParser()
    valid_run_names = ['wisconsin', 'california', 'eg', 'hs']
    valid_task_names = ['send', 'resend_failed', 'resend_invalid', 'retrieve', 'finalize']
    parser.add_argument('-r', '--run_name', type=str, help=f"RUN_NAME can be: {', '.join(valid_run_names)}.", required=True)
    parser.add_argument('-t', '--task_name', type=str, help=f"TASK_NAME can be: {', '.join(valid_task_names)}.", required=True)
    args = parser.parse_args()

    run_name = args.run_name
    if run_name not in valid_run_names:
        raise ValueError('Invalid run_name argument.')
    task_name = args.task_name
    if task_name not in valid_task_names:
        raise ValueError('Invalid task_name argument.')

    from is_gpt_bayesian.utils import time_utils, path_utils
    import logging.config

    run_path = path_utils.run_path(run_name)
    path_utils.create_path(run_path)
    log_path = path_utils.log_path(run_name)

    logging.config.fileConfig('logging.conf', defaults={'logfilename': log_path})
    logger = logging.getLogger(__name__)
    logger.info(f'RUN_NAME: {run_name}, TASK_NAME: {task_name}.')


    # ===================================
    # Import modules
    # ===================================

    import numpy as np
    import pandas as pd
    from is_gpt_bayesian.model import OpenAISession
    from is_gpt_bayesian.processing import (specs_processing, 
                                            prompt_processing, 
                                            response_processing)


    # ===================================
    # Model Settings
    # ===================================

    # Temperatures
    temperature_lower_bound = 1
    temperature_upper_bound = 1

    # Models
    models = [
            # "gpt-4o",                 # points to gpt-4o-2024-08-06
            # "gpt-4o-mini",            # points to gpt-4o-mini-2024-07-18
            # "gpt-4",                  # points to gpt-4-0613
            # "gpt-4-turbo",            # points to gpt-4-turbo-2024-04-09
            # "gpt-3.5-turbo-0125",     # points to gpt-3.5-turbo-0125
            "o1-mini",                # points to o1-mini-2024-09-12
            "o1-preview"              # points to o1-preview-2024-09-12
            ]

    # Instructions
    instructions = ['reasoning',
                    'no reasoning']

    # Model seed
    seeds = []


    # ===================================
    # Generating specs
    # ===================================

    # El-Gamal and Grether
    if run_name == 'california':

        if task_name in ['send', 'resend_invalid']:
            run_specs = specs_processing.get_california_specs_df(temperature_lower_bound, temperature_upper_bound,
                                                                 models,
                                                                 instructions,
                                                                 seeds)
        response_fnc = response_processing.response_eg

    elif run_name == 'wisconsin':

        if task_name in ['send', 'resend_invalid']:
            run_specs = specs_processing.get_wisconsin_specs_df(temperature_lower_bound, temperature_upper_bound,
                                                                models,
                                                                instructions,
                                                                seeds)
        response_fnc = response_processing.response_eg

    elif run_name == 'hs':

        if task_name in ['send', 'resend_invalid']:
            run_specs = specs_processing.get_hs_specs_df(temperature_lower_bound, temperature_upper_bound,
                                                        models,
                                                        instructions,
                                                        seeds)

        response_fnc = response_processing.response_hs

    else:

        raise ValueError('Invalid run_name argument.')


    # ===================================
    # Run task
    # ===================================

    try:
        # run_specs = run_specs.head(4)
        # run_specs = run_specs[(run_specs['subject_id']=='DATA11&12 - Subject 1') &
        #                       (run_specs['instruction']=='reasoning')]
        run_specs = run_specs[(run_specs['subject_id']=='Part 1 Holt and Smith - id S1') &
                        (run_specs['instruction']=='reasoning')]
        print(run_specs)
    except:
        pass

    if task_name == 'send':

        session = OpenAISession(run_name)
        session.generate_batch_files(run_specs)
        session.send_batches()

    elif task_name == 'resend_failed':

        session = OpenAISession(run_name)
        session.resend_failed_jobs()

    elif task_name == 'resend_invalid':

        run_specs = pd.read_csv(path_utils.run_final_stacked_file_path(run_name), index_col=0)
        run_specs = run_specs[(run_specs['processed_response'].isna()) &
                              (run_specs['query_idx'] == run_specs['query_total_count'])]
        
        if len(run_specs) == 0:
            logger.info('ALL JOBS ARE COMPLETED. NO INVALID PROCESSED_RESPONSE.')
        else:
            specs_cols = pd.read_csv(path_utils.run_specs_file_path(run_name), index_col=0, nrows=0).columns
            session = OpenAISession(run_name)
            session.generate_batch_files(run_specs[specs_cols])
            session.send_batches()

    elif task_name == 'retrieve': 

        session = OpenAISession(run_name)
        session.load_jobs()
        session.retrieve_batches()
        session.process_reponses()

    elif task_name == 'finalize':

        session = OpenAISession(run_name)
        session.load_jobs()
        session.retrieve_batches()
        results_df = session.process_reponses()

        if run_name in ['california', 'wisconsin']:
            final_df_stacked_dict, final_df_unstacked_ungrouped_dict = response_processing.process_eg_result_df(results_df, response_fnc, run_name, ungroup_by=['name'])
        elif run_name == 'hs':
            final_df_stacked_dict, final_df_unstacked_ungrouped_dict = response_processing.process_hs_result_df(results_df, response_fnc, run_name, ungroup_by=['sheet_name'])
        
        for path, final_df in final_df_stacked_dict.items():
            final_df_stacked = final_df
            final_df.to_csv(path)
            logger.info(f'Stacked run results df saved to {path}.')
        
        for path, final_df in final_df_unstacked_ungrouped_dict.items():
            if isinstance(final_df.index, pd.MultiIndex):
                final_df.to_csv(path)
            else:
                final_df.to_csv(path, index=False)
            logger.info(f'Processed / unstacked ungrouped run results df saved to {path}.')

        # check for invalid
        results_df_invalid = final_df_stacked[(final_df_stacked['processed_response'].isna()) &
                                              (final_df_stacked['query_idx'] == final_df_stacked['query_total_count'])]
        
        if len(results_df_invalid) == 0:
            logger.info('Run result df has NO invalid processed_response.')
        else:
            logger.info(f'Run result df has {len(results_df_invalid)} invalid processed_response.')

    else:

        raise ValueError('Invalid task_name argument.')
