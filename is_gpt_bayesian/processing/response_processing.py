import re
from is_gpt_bayesian.utils import time_utils, path_utils


answer_A = "Cage A".replace(' ', '').lower()
answer_B = "Cage B".replace(' ', '').lower()


def response_eg(textual_response):
    processed_response = textual_response.split('\n')[-1].replace(' ', '').lower()
    if answer_A in processed_response and answer_B in processed_response:
        return None
    elif answer_A in processed_response:
        return '1'
    elif answer_B in processed_response:
        return '0'
    else:
        return None


def response_hs(textual_response):

    # 1. Find the last non-empty line
    lines = textual_response.strip().split('\n')
    last_non_empty_line = None
    for line in reversed(lines):
        # Strip whitespace and see if anything remains
        stripped = line.strip()
        if stripped:
            last_non_empty_line = stripped
            break

    if last_non_empty_line is None:
        return None

    # 2. Check if that line contains 'final answer' (case-insensitive)
    if not re.search(r'final answer', last_non_empty_line, re.IGNORECASE):
        return None

    # 3. Extract fraction, decimal, or integer
    fraction_match = re.search(r'(\d+\s*/\s*\d+)', last_non_empty_line)
    if fraction_match:
        fraction_str = fraction_match.group(1)
        fraction_str = fraction_str.replace(' ', '')
        numerator, denominator = fraction_str.split('/')
        try:
            return float(numerator) / float(denominator)
        except ZeroDivisionError:
            return None

    decimal_match = re.search(r'\d+\.\d+', last_non_empty_line)
    if decimal_match:
        return float(decimal_match.group(0))

    int_match = re.search(r'\d+', last_non_empty_line)
    if int_match:
        return float(int_match.group(0))

    # 5. If no pattern found, return None
    return None


def process_result_df(results_df, response_processing_fnc, run_name, ungroup_by):
    # stacked
    results_df_stacked = results_df.copy()
    results_df_stacked['processed_response'] = results_df_stacked['textual_response'].apply(response_processing_fnc)
    try:
        if results_df_stacked['processed_response'] == results_df_stacked['processed_response'].astype(int).astype(float):
            results_df_stacked['processed_response'] = results_df_stacked['processed_response'].astype(int)
    except:
        results_df_stacked['processed_response'] = results_df_stacked['processed_response'].astype(float)
    
    # unstacked
    results_df_last_query = results_df_stacked[results_df_stacked['query_idx'] == results_df_stacked['query_total_count']]
    
    columns_name_list = ['subject_id', 'subject_uuid', 'temperature']
    values_name_list = ['processed_response']
    del_name_list = ['obs_idx', 'batch_id', 'prompt', 'request_id', 'textual_response', 'created_time', 'query_idx', 'query_total_count']		

    if ungroup_by:

        unstacked_ungrouped_results_df_dict = {}

        for group_name, group_results_df in results_df_last_query.groupby(ungroup_by):
            results_df_unstacked_ungrouped_filename = path_utils.run_final_unstacked_ungrouped_file_path(run_name, '__'.join(group_name))
            results_df_unstacked = group_results_df.pivot(
                index=[col for col in group_results_df.columns if col not in columns_name_list + values_name_list + del_name_list],
                columns=columns_name_list, 
                values=values_name_list)
            unstacked_ungrouped_results_df_dict[results_df_unstacked_ungrouped_filename] = results_df_unstacked
        
        return ({path_utils.run_final_stacked_file_path(run_name): results_df_stacked}, 
                unstacked_ungrouped_results_df_dict
                )

    else:
        results_df_unstacked = results_df_last_query.pivot(
            index=[col for col in results_df_last_query.columns if col not in columns_name_list + values_name_list + del_name_list],
            columns=columns_name_list, 
            values=values_name_list)
        
        return ({path_utils.run_final_stacked_file_path(run_name): results_df_stacked}, 
                {path_utils.run_final_unstacked_file_path(run_name): results_df_unstacked}
                )