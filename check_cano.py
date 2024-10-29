from utils.preprocess_smi import canonicalize_smiles
import pandas as pd

ori_df = pd.read_csv('data/uspto_50k/raw_val.csv')
prprocessed_df = pd.read_csv('preprocessed/debug_uspto_50k/val.csv')

ori_reactants = [canonicalize_smiles(rxn.split('>>')[0]) for rxn in ori_df['reactants>reagents>production']]
preprocessed_reactants = [react.replace(' ', '') for react in prprocessed_df['substrates_token']]

demo_ori_react = ori_reactants[0]
demo_preprocessed_react = preprocessed_reactants[0]

assert set(ori_reactants) == set(preprocessed_reactants) and demo_preprocessed_react == demo_ori_react





