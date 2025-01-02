import re
from is_gpt_bayesian.utils import time_utils, path_utils
import unicodedata


NUM_WORDS = {
    "ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5, "SIX": 6,
    "SEVEN": 7, "EIGHT": 8, "NINE": 9, "TEN": 10, "ELEVEN": 11, "TWELVE": 12
}

DEN_WORDS = {
    "HALF": 2, "HALVES": 2,
    "THIRD": 3, "THIRDS": 3,
    "FOURTH": 4, "FOURTHS": 4,
    "FIFTH": 5, "FIFTHS": 5,
    "SIXTH": 6, "SIXTHS": 6,
    "SEVENTH": 7, "SEVENTHS": 7,
    "EIGHTH": 8, "EIGHTHS": 8,
    "NINTH": 9, "NINTHS": 9,
    "TENTH": 10, "TENTHS": 10,
    "ELEVENTH": 11, "ELEVENTHS": 11,
    "TWELFTH": 12, "TWELFTHS": 12
}

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
        if "equal" in processed_response or "indifferent" in processed_response:
            return '0.5'
        return None

def parse_single_char_fraction(ch: str) -> float | None:
    """
    Tries to parse a single Unicode 'vulgar fraction' character (e.g., ½, ⅓, ⅔, ⅘).
    Returns the corresponding float if recognized, or None if it cannot be parsed.
    """
    if len(ch) != 1:
        return None  # Must be exactly one character

    # Get the official Unicode name, e.g. "VULGAR FRACTION ONE HALF", "VULGAR FRACTION TWO THIRDS"
    try:
        name = unicodedata.name(ch)
    except ValueError:
        # Character has no name in Unicode or is invalid
        return None

    # Check if it is indeed a "VULGAR FRACTION ..."
    # e.g. name might be: "VULGAR FRACTION THREE EIGHTHS"
    if not name.startswith("VULGAR FRACTION "):
        return None

    # Remove the leading "VULGAR FRACTION " (16 characters)
    remainder = name[len("VULGAR FRACTION "):]  # e.g., "THREE EIGHTHS"

    # Split into words, e.g. ["THREE", "EIGHTHS"]
    words = remainder.split()

    # Typically, this will be 2 words for standard fractions: [NUMERATOR_WORD, DENOMINATOR_WORD],
    # e.g. ["TWO", "THIRDS"] -> 2/3, ["ONE", "HALF"] -> 1/2, ["THREE", "EIGHTHS"] -> 3/8, etc.
    #
    # There *are* some less-common multi-word sequences like "ONE HUNDRED TWENTY-EIGHTH",
    # but for typical usage we can assume numerator is the first word, denominator is the second.
    if len(words) != 2:
        return None  # We won't handle multi-word fraction names beyond the typical ones.

    num_word, den_word = words  # e.g. "THREE", "EIGHTHS"

    # Convert numerator word -> integer
    numerator = NUM_WORDS.get(num_word)
    if numerator is None:
        return None

    # Convert denominator word -> integer
    denominator = DEN_WORDS.get(den_word)
    if denominator is None:
        return None

    try:
        return numerator / denominator
    except ZeroDivisionError:
        return None
    

def response_hs(textual_response):
    """
    1) Find the LAST occurrence of "final answer" in the text (case-insensitive).
    2) Extract everything AFTER that phrase.
    3) Parse that substring for:
       - LaTeX fraction  (\dfrac{2.5}{3.5} or \frac{1}{2})
       - Plain fraction  (e.g. "2.5 / 3.5" or "2⁄3")
       - Decimal         (e.g. "0.5")
       - Integer         (e.g. "2")
       - Single-char fraction (e.g. "½", "⅔")
    4) Return the float value, or None if parsing fails.
    """

    if not isinstance(textual_response, str):
        return None

    # ---------------------------------------------------------
    # 1) Find the last occurrence of "final answer" (case-insensitive)
    # ---------------------------------------------------------
    matches = list(re.finditer(r'final answer', textual_response, re.IGNORECASE))
    if not matches:
        return None  # No occurrence at all

    last_match = matches[-1]
    # We'll parse everything *after* "final answer"
    start_pos = last_match.end()
    substring = textual_response[start_pos:].strip()

    # If nothing after 'final answer', we might parse from the same line 
    # in case the fraction or number is on that same line.
    if not substring:
        substring = textual_response[last_match.start():].strip()
        if not substring:
            return None

    # This is our candidate string to parse
    line_of_interest = substring

    # ---------------------------------------------------------
    # 2a) LaTeX fraction: \dfrac{2}{3} or \frac{2.5}{3.5}
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 2b) Plain fraction with optional decimals, using '/' or '⁄'
    # ---------------------------------------------------------
    fraction_pattern = re.compile(
        r'([0-9]*\.?[0-9]+)\s*[/⁄]\s*([0-9]*\.?[0-9]+)'
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

    # ---------------------------------------------------------
    # 2c) Decimal
    # ---------------------------------------------------------
    decimal_match = re.search(r'\d+\.\d+', line_of_interest)
    if decimal_match:
        try:
            return float(decimal_match.group(0))
        except ValueError:
            return None

    # ---------------------------------------------------------
    # 2d) Integer
    # ---------------------------------------------------------
    int_match = re.search(r'\d+', line_of_interest)
    if int_match:
        try:
            return float(int_match.group(0))
        except ValueError:
            return None

    # ---------------------------------------------------------
    # 2e) Single-character fraction (vulgar fractions)
    # ---------------------------------------------------------
    # We look for any single non-whitespace character, and test if it's a fraction.
    # E.g. "½", "⅓", "⅔", etc. We'll scan all single chars in the substring.
    # If we find a recognized fraction, we return it.
    for ch in line_of_interest:
        if ch.isspace():
            continue
        fraction_value = parse_single_char_fraction(ch)
        if fraction_value is not None:
            return fraction_value

    # ---------------------------------------------------------
    # If none matched, return None
    # ---------------------------------------------------------
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
    results_df_stacked['posterior_prob'] = results_df_stacked.apply(eg_posterior_probability, axis=1)
    
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
            {path_utils.run_final_stacked_processed_file_path(run_name): results_df_stacked_processed}
            )


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