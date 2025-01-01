import pandas as pd
import numpy as np
import hashlib
from scipy.io import loadmat
from is_gpt_bayesian.utils import time_utils, path_utils
from is_gpt_bayesian.processing import prompt_processing

def md5_hash(s) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)

def random_uniform_on_grid(seed, grid) -> float:
    rng = np.random.default_rng(seed)
    return rng.choice(grid)


def get_california_data() -> pd.DataFrame:

    data_eg_path = path_utils.data_eg_path

    data = loadmat(data_eg_path)
    data = data['datastruct'][0]

    specs_list = []

    for design in data:
        if design['state'][0] != 'california':
            continue
        specs_dict = {}
        specs_dict['name'] = design['name'][0]
        specs_dict['priors'] = design['priors'].squeeze()
        specs_dict['ndraws'] = design['ndraws'].squeeze()
        specs_dict['nsubjects'] = design['subjectchoices'].shape[0]
        specs_dict['ntrials'] = design['subjectchoices'].shape[1]
        specs_dict['pay'] = design['pay'][0][0]
        specs_dict['nballs'] = design['nballs'][0][0]
        specs_dict['ndraws_from_cage'] = design['ndraws_from_cage'][0][0]
        specs_dict['cage_A_balls_marked_N'] = design['cage_A_balls_marked_N'][0][0]
        specs_dict['cage_B_balls_marked_N'] = design['cage_B_balls_marked_N'][0][0]
        specs_dict['nballs_prior_cage'] = design['nballs_prior_cage'][0][0]
        specs_dict['state'] = design['state'][0]

        design_df = pd.DataFrame.from_dict(specs_dict)
        design_df['trial_id'] = np.arange(1, specs_dict['ntrials'] + 1)
        design_df['trial_id'] = design_df['name'] + ' - Trial ' + design_df['trial_id'].astype(str)
        design_df = pd.concat([design_df] * specs_dict['nsubjects'], ignore_index=True)
        design_df['subject_id'] = np.repeat(np.arange(1, specs_dict['nsubjects'] + 1), repeats=specs_dict['ntrials'])
        design_df['subject_id'] = design_df['name'] + ' - Subject ' + design_df['subject_id'].astype(str)
        specs_list.append(design_df) 

    specs_df = pd.concat(specs_list, ignore_index=True)

    specs_df = pd.concat([pd.DataFrame({'obs_idx': range(len(specs_df))}),
                          specs_df], axis=1)
    
    specs_df = specs_df[['obs_idx', 'name', 'state', 'trial_id', 'subject_id', 'nsubjects', 'ntrials', 'pay', 'nballs', 
                        'ndraws_from_cage', 'cage_A_balls_marked_N', 'cage_B_balls_marked_N', 'nballs_prior_cage', 
                        'priors', 'ndraws']]
    
    specs_df['subject_uuid'] = specs_df['subject_id'].apply(md5_hash)
    
    return specs_df


def get_wisconsin_data() -> pd.DataFrame:

    data_eg_path = path_utils.data_eg_path

    data = loadmat(data_eg_path)
    data = data['datastruct'][0]

    specs_list = []

    for design in data:
        if design['state'][0] != 'wisconsin':
            continue
        specs_dict = {}
        specs_dict['name'] = design['name'][0]
        specs_dict['priors'] = design['priors'].squeeze()
        specs_dict['ndraws'] = design['ndraws'].squeeze()
        specs_dict['nsubjects'] = design['subjectchoices'].shape[0]
        specs_dict['ntrials'] = design['subjectchoices'].shape[1]
        specs_dict['pay'] = design['pay'][0][0]
        specs_dict['nballs'] = design['nballs'][0][0]
        specs_dict['ndraws_from_cage'] = design['ndraws_from_cage'][0][0]
        specs_dict['cage_A_balls_marked_N'] = design['cage_A_balls_marked_N'][0][0]
        specs_dict['cage_B_balls_marked_N'] = design['cage_B_balls_marked_N'][0][0]
        specs_dict['nballs_prior_cage'] = design['nballs_prior_cage'][0][0]
        specs_dict['state'] = design['state'][0]

        design_df = pd.DataFrame.from_dict(specs_dict)
        design_df['trial_id'] = np.arange(1, specs_dict['ntrials'] + 1)
        design_df['trial_id'] = design_df['name'] + ' - Trial ' + design_df['trial_id'].astype(str)
        design_df = pd.concat([design_df] * specs_dict['nsubjects'], ignore_index=True)
        design_df['subject_id'] = np.repeat(np.arange(1, specs_dict['nsubjects'] + 1), repeats=specs_dict['ntrials'])
        design_df['subject_id'] = design_df['name'].map({'DATA11': 'DATA11&12',
                                                         'DATA12': 'DATA11&12',
                                                         'DATA21': 'DATA21&22',
                                                         'DATA22': 'DATA21&22'}) + ' - Subject ' + design_df['subject_id'].astype(str)
        specs_list.append(design_df) 

    specs_df = pd.concat(specs_list, ignore_index=True)

    specs_df = pd.concat([pd.DataFrame({'obs_idx': range(len(specs_df))}),
                          specs_df], axis=1)
    
    specs_df = specs_df[['obs_idx', 'name', 'state', 'trial_id', 'subject_id', 'nsubjects', 'ntrials', 'pay', 'nballs', 
                        'ndraws_from_cage', 'cage_A_balls_marked_N', 'cage_B_balls_marked_N', 'nballs_prior_cage', 
                        'priors', 'ndraws']]
    
    specs_df['subject_uuid'] = specs_df['subject_id'].apply(md5_hash)
    
    return specs_df


def get_hs_data() -> pd.DataFrame:

    holt_and_smith_data_path = path_utils.data_hs_path
    def _read_data_helper(holt_and_smith_data_path, sheet_name):
        df = pd.read_excel(holt_and_smith_data_path, sheet_name=sheet_name)
        df = pd.melt(
                    df,
                    id_vars=df.columns[:2],
                    value_vars=df.columns[2:], 
                    var_name='id',
                    value_name='outcome' 
                )
        df['Prior Pr(A)'] = df['Prior Pr(A)'].str.replace(' ', '').map({'1/2': '1/2',
                                                                        '½': '1/2', 
                                                                        '2/3': '2/3',
                                                                        r'\2/3': '2/3'})
        df['prior'] = df['Prior Pr(A)'].map({'1/2': 1/2,
                                             '2/3': 2/3})
        df['outcome'] = df['outcome'].str.replace(r'[^dDlL]', '', regex=True).str.upper()
        df = df[df['outcome'] != '']
        df['ndraws_from_cage'] = df['outcome'].str.len()
        df['D_draws_from_cage'] = df['outcome'].str.count('D')
        df['L_draws_from_cage'] = df['outcome'].str.count('L')
        df['outcome_expand'] = df['outcome'].str.replace('D', 'Dark, ').str.replace('L', 'Light, ').str[:-2]
        df['sheet_name'] = sheet_name
        df['trial_id'] = df['sheet_name'] + ' - Round ' + df['Round'].astype(str)
        df['subject_id'] = df['sheet_name'] + ' - id ' + df['id'].astype(str)
        
        return df

    df1 = _read_data_helper(holt_and_smith_data_path, 'Part 1 Holt and Smith')
    df2 = _read_data_helper(holt_and_smith_data_path, 'Part 2 Holt and Smith')
    df3 = _read_data_helper(holt_and_smith_data_path, 'Part 3 Holt and Smith')
    df4 = _read_data_helper(holt_and_smith_data_path, 'Part 4 Holt and Smith')

    specs_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

    specs_df = pd.concat([pd.DataFrame({'obs_idx': range(len(specs_df))}),
                          specs_df], axis=1)
    
    specs_df = specs_df[['obs_idx', 'sheet_name', 'trial_id', 'subject_id', 'Prior Pr(A)', 'prior', 'outcome', 'ndraws_from_cage', 'D_draws_from_cage', 'L_draws_from_cage', 'outcome_expand']]

    specs_df['subject_uuid'] = specs_df['subject_id'].apply(md5_hash)
    
    return specs_df


def _get_specs_df(data_df, 
                 temperature_lower_bound, temperature_upper_bound,
                 models,
                 instructions,
                 seeds,
                 prompt_fnc) -> pd.DataFrame:
    
    data_df = data_df.copy()
    grid = np.arange(temperature_lower_bound, temperature_upper_bound + 0.001, 0.01) # using +0.001 to avoid floating point error
    data_df['temperature'] = data_df['subject_uuid'].apply(lambda seed: random_uniform_on_grid(seed, grid))
    
    # Cartiesian product with models
    data_df = pd.merge(data_df,
                       pd.DataFrame(models, columns=['model']),
                       how='cross')
    # Cartiesian product with instructions
    data_df = pd.merge(data_df,
                       pd.DataFrame(instructions, columns=['instruction']),
                       how='cross')
    # Cartiesian product with seeds
    if seeds:
        data_df = pd.merge(data_df,
                        pd.DataFrame(seeds, columns=['seed']),
                        how='cross')
    # Add prompt column
    data_df['prompt'] = data_df.apply(prompt_fnc, axis=1)

    return data_df


def get_california_specs_df(temperature_lower_bound, temperature_upper_bound,
                    models,
                    instructions,
                    seeds) -> pd.DataFrame:

    data_df = get_california_data()
    data_df = _get_specs_df(data_df,
                           temperature_lower_bound, temperature_upper_bound,
                           models,
                           instructions,
                           seeds,
                           prompt_processing.prompt_eg)

    return data_df


def get_wisconsin_specs_df(temperature_lower_bound, temperature_upper_bound,
                    models,
                    instructions,
                    seeds) -> pd.DataFrame:

    data_df = get_wisconsin_data()
    data_df = _get_specs_df(data_df,
                           temperature_lower_bound, temperature_upper_bound,
                           models,
                           instructions,
                           seeds,
                           prompt_processing.prompt_eg)

    return data_df


def get_hs_specs_df(temperature_lower_bound, temperature_upper_bound,
                    models,
                    instructions,
                    seeds) -> pd.DataFrame:

    data_df = get_hs_data()
    data_df = _get_specs_df(data_df,
                           temperature_lower_bound, temperature_upper_bound,
                           models,
                           instructions,
                           seeds,
                           prompt_processing.prompt_hs)

    return data_df