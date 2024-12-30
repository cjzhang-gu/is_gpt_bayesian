import re
from is_gpt_bayesian.utils import time_utils, path_utils


answer_A = "Cage A".replace(' ', '').lower()
answer_B = "Cage B".replace(' ', '').lower()


def response_eg(textual_response):
    if not isinstance(textual_response, str):
        return None
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
    """
    Looks for a 'final answer' line (case-insensitive).
    Then attempts to parse, in order:
      1) LaTeX fraction  ( e.g. \dfrac{2.5}{3.5} )
      2) Plain fraction  ( e.g. 2.5 / 3.5 )
      3) Decimal         ( e.g. 2.5 )
      4) Integer         ( e.g. 2 )
    Returns float or None if none is found.
    """

    if not isinstance(textual_response, str):
        return None

    # 1. Find the last non-empty line containing anything
    lines = textual_response.strip().split('\n')
    last_non_empty_line = None
    for line in reversed(lines):
        if line.strip():
            last_non_empty_line = line.strip()
            break
    
    if last_non_empty_line is None:
        return None

    # 2. Check if that line contains 'final answer' (case-insensitive)
    if not re.search(r'final answer', last_non_empty_line, re.IGNORECASE):
        return None

    # We now attempt to parse from that line. Let's call it line_of_interest
    line_of_interest = last_non_empty_line

    # ----------------------------------------------------------------------
    # 3a. Look for a LaTeX fraction: \dfrac{2}{3} or \frac{2.5}{3.5}, etc.
    #     Pattern:  \d?frac{ <num> }{ <num> }
    #
    #     Group 1 captures the numerator
    #     Group 2 captures the denominator
    #
    #     We allow digits with optional decimal point in numerator/denominator.
    # ----------------------------------------------------------------------
    latex_fraction_pattern = re.compile(
        r'\\d?frac\s*\{\s*([0-9]*\.?[0-9]+)\s*\}\s*\{\s*([0-9]*\.?[0-9]+)\s*\}',
        re.IGNORECASE
    )

    latex_fraction_match = latex_fraction_pattern.search(line_of_interest)
    if latex_fraction_match:
        numerator_str, denominator_str = latex_fraction_match.groups()
        try:
            num = float(numerator_str)
            den = float(denominator_str)
            return num / den
        except (ZeroDivisionError, ValueError):
            return None

    # ----------------------------------------------------------------------
    # 3b. Look for a plain fraction with optional decimals: e.g. 2/3 or 2.5 / 3.5
    # ----------------------------------------------------------------------
    fraction_pattern = re.compile(
        r'([0-9]*\.?[0-9]+)\s*/\s*([0-9]*\.?[0-9]+)'
    )
    fraction_match = fraction_pattern.search(line_of_interest)
    if fraction_match:
        numerator_str, denominator_str = fraction_match.groups()
        try:
            num = float(numerator_str)
            den = float(denominator_str)
            return num / den
        except (ZeroDivisionError, ValueError):
            return None

    # ----------------------------------------------------------------------
    # 3c. Look for a decimal: e.g. 0.5
    # ----------------------------------------------------------------------
    decimal_match = re.search(r'\d+\.\d+', line_of_interest)
    if decimal_match:
        try:
            return float(decimal_match.group(0))
        except ValueError:
            return None

    # ----------------------------------------------------------------------
    # 3d. Look for an integer: e.g. 1
    # ----------------------------------------------------------------------
    int_match = re.search(r'\d+', line_of_interest)
    if int_match:
        try:
            return float(int_match.group(0))
        except ValueError:
            return None

    # ----------------------------------------------------------------------
    # 4. If no pattern found, return None
    # ----------------------------------------------------------------------
    return None


def process_eg_result_df(results_df, response_processing_fnc, run_name, ungroup_by):
    # stacked
    results_df_stacked = results_df.copy()
    results_df_stacked['processed_response'] = results_df_stacked['textual_response'].apply(response_processing_fnc)
    try:
        if results_df_stacked['processed_response'] == results_df_stacked['processed_response'].astype(int).astype(float):
            results_df_stacked['processed_response'] = results_df_stacked['processed_response'].astype(int)
    except:
        results_df_stacked['processed_response'] = results_df_stacked['processed_response'].astype(float)

    # adding output columns
    # results_df_stacked['posterior_prob'] = results_df_stacked.apply(eg_posterior_probability, axis=1)
    
    # unstacked
    results_df_last_query = results_df_stacked[results_df_stacked['query_idx'] == results_df_stacked['query_total_count']]
    
    columns_name_list = ['subject_id', 'subject_uuid', 'temperature']
    values_name_list = ['processed_response']
    del_name_list = ['obs_idx', 'batch_id', 'prompt', 'request_id', 'textual_response', 'created_time', 'query_idx', 'query_total_count']		

    if ungroup_by:

        unstacked_ungrouped_results_df_dict = {}

        for group_name, group_results_df in results_df_last_query.groupby(ungroup_by):
            results_df_unstacked_ungrouped_filename = path_utils.run_final_unstacked_ungrouped_file_path(run_name, '__'.join(group_name))
            results_df_unstacked_ungrouped_mat_filename = path_utils.run_final_unstacked_ungrouped_mat_file_path(run_name, '__'.join(group_name))
            results_df_unstacked_ungrouped_subject_filename = path_utils.run_final_unstacked_ungrouped_subject_file_path(run_name, '__'.join(group_name))
            # full info pivot
            results_df_unstacked = group_results_df.pivot(
                index=[col for col in group_results_df.columns if col not in columns_name_list + values_name_list + del_name_list],
                columns=columns_name_list, 
                values=values_name_list)
            # mat value
            results_df_unstacked_mat = results_df_unstacked.copy()
            results_df_unstacked_mat.columns = results_df_unstacked_mat.columns.get_level_values('subject_id').rename(None)
            results_df_unstacked_mat = results_df_unstacked_mat.reset_index()
            # subject info
            results_df_unstacked_subject = results_df_unstacked.columns.to_frame().reset_index(drop=True).iloc[:, 1:]
            unstacked_ungrouped_results_df_dict[results_df_unstacked_ungrouped_filename] = results_df_unstacked
            unstacked_ungrouped_results_df_dict[results_df_unstacked_ungrouped_mat_filename] = results_df_unstacked_mat
            unstacked_ungrouped_results_df_dict[results_df_unstacked_ungrouped_subject_filename] = results_df_unstacked_subject
        
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
    

def process_hs_result_df(results_df, response_processing_fnc, run_name, ungroup_by):
    # stacked
    results_df_stacked = results_df.copy()
    results_df_stacked['processed_response'] = results_df_stacked['textual_response'].apply(response_processing_fnc)
    try:
        if results_df_stacked['processed_response'] == results_df_stacked['processed_response'].astype(int).astype(float):
            results_df_stacked['processed_response'] = results_df_stacked['processed_response'].astype(int)
    except:
        results_df_stacked['processed_response'] = results_df_stacked['processed_response'].astype(float)

    # adding output columns
    results_df_stacked['posterior_prob'] = results_df_stacked.apply(hs_posterior_probability, axis=1)

    # processed
    results_df_stacked_processed = results_df_stacked[results_df_stacked['query_idx'] == results_df_stacked['query_total_count']]
    results_df_stacked_processed = results_df_stacked_processed.sort_values(['model', 'obs_idx', 'instruction'])

    return ({path_utils.run_final_stacked_file_path(run_name): results_df_stacked},
            {path_utils.run_final_stacked_processed_file_path(run_name): results_df_stacked_processed})


def eg_posterior_probability(row):
    likelihood = (row['cage_A_balls_marked_N']/row['nballs']) ** row['ndraws'] * \
                 (1 - row['cage_A_balls_marked_N']/row['nballs']) ** (row['ndraws_from_cage'] - row['ndraws'])
    
    prior = row['priors'] / row['nballs_prior_cage']

    marginal = likelihood * prior + \
               (row['cage_B_balls_marked_N']/row['nballs']) ** row['ndraws'] * \
               (1 - row['cage_B_balls_marked_N']/row['nballs']) ** (row['ndraws_from_cage'] - row['ndraws']) * (1-prior)
    
    return likelihood * prior / marginal


def hs_posterior_probability(row):
    A_light_prob = 2 / 3
    B_light_prob = 1 / 3
    likelihood = (A_light_prob ** row['L_draws_from_cage']) * \
                 ((1-A_light_prob) ** row['D_draws_from_cage'])
    
    if row['Prior Pr(A)'] == '1/2':
        prior = 1 / 2
    elif row['Prior Pr(A)'] == '2/3':
        prior = 2 / 3
    
    marginal = likelihood * prior + \
               (B_light_prob ** row['L_draws_from_cage']) * \
               ((1-B_light_prob) ** row['D_draws_from_cage']) * (1-prior)
    
    return likelihood * prior / marginal