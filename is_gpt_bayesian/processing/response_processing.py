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
    return response_eg(textual_response)


def process_result_df(results_df, response_processing_fnc):
    # stacked
    results_df_stacked = results_df.copy()
    results_df_stacked['processed_response'] = results_df_stacked['textual_response'].apply(response_processing_fnc)
    
    # unstacked
    results_df_last_query = results_df_stacked[results_df_stacked['query_idx'] == results_df_stacked['query_total_count']]
    
    columns_name_list = ['subject_id', 'subject_uuid', 'temperature']
    values_name_list = ['processed_response']
    del_name_list = ['batch_id', 'prompt', 'request_id', 'textual_response']
    
    results_df_unstacked = results_df_last_query.pivot(
        index=[col for col in results_df_last_query.columns if col not in columns_name_list + values_name_list + del_name_list],
        columns=columns_name_list, 
        values=values_name_list)
    
    return results_df_stacked, results_df_unstacked