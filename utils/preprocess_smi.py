import re
from rdkit import Chem
from typing import Tuple
import os
import collections
import numpy as np
from tqdm import tqdm
from typing import Dict,List
from matplotlib import pyplot as plt

from .dataset_basic import DataLoad


def canonicalize_smiles(smi: str, map_clear=True, cano_with_heavyatom=True) -> str:
    cano_smi = ''
    mol = Chem.MolFromSmiles(smi)

    if mol is None:
        cano_smi = ''
    else:
        if mol.GetNumHeavyAtoms() < 2 and cano_with_heavyatom:
            cano_smi = 'CC'
        elif map_clear:
            for a in mol.GetAtoms():
                a.ClearProp('molAtomMapNumber')
            cano_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        else:
            cano_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
    return cano_smi


def hydro_remove(subs: str, prod: str) -> Tuple[str, str]:
    subs = re.sub(r'\[(?P<atom>[CNOc])H[0-9]?\]', r'\g<atom>', subs)
    prod = re.sub(r'\[(?P<atom>[CNOc])H[0-9]?\]', r'\g<atom>', prod)

    #atom_index = set([atom for atom in re.findall(r'[a-zA-Z][a-zA-Z]?', subs+'>>'+prod)])
    #symbol_index = set([symbol for symbol in re.findall(r'[^a-zA-Z>>]', subs+'>>'+prod)])

    #return subs, prod, list(atom_index), list(symbol_index)
    return subs, prod


def space_remove(atom_match):
    #print(atom_match)
    return atom_match.group('atom').replace(' ', '')


def atom_space_remove(atom_match):
    match = atom_match.group('atom')
    pattern = r'(?P<atom>[A-z][\s][a-z])'
    match = re.sub(pattern, space_remove, match)
    return match


def space_add(atom_match):
    return ' '.join(atom_match.group('atom'))


def atom_split(subs: str, prod: str) -> Tuple[str, str]:
    plural_letter_atom = (
        r'(?P<atom>H\se|L\si|B\se|N\se|N\sa|M\sg|A\sl|S\si|C\sl|A\sr'
        r'|C\sa|S\sc|T\si|C\sr|M\sn|F\se|C\so|N\si|C\su|Z\sn|G\sa|G\se|A\ss|S\se|B\sr|K\sr|R\sb|S\sr|Z\sr|N\sb|M\so|T\sc|R\su|R\sh|P\sd|A\sg|C\sd|I\sn|S\sn|S\sb|T\se|X\se'
        r'|C\ss|B\sa|L\sa|C\se|P\sr|N\sd|P\sm|S\sm|E\su|G\sd|T\sb|D\sy|H\so|E\sr|T\sm|Y\sb|L\su|H\sf|T\sa|R\se|O\ss|I\sr|P\st|A\su|H\sg|T\si|P\sb|B\si|P\so|A\st|R\sn'
        r'|F\sr|R\sa|A\sc|T\sh|P\sa|N\sp|P\su|A\sm|C\sm|B\sk|C\sf|E\ss|F\sm|M\sd|N\so|L\sr|R\sf|D\sb|S\sg|B\sh|H\ss|M\st|D\ss|R\sg|C\sn|N\sh|F\sl|M\sc|L\sv|T\ss|O\sg)'
    )

    IIIA_to_0 = r'(?P<atom>H\se|N\se|A\sl|S\si|C\sl|A\sr|G\sa|G\se|A\ss|S\se|B\sr|K\sr|I\sn|S\sn|S\sb|T\se|X\se|T\sl|P\sb|B\si|P\so|A\st|R\sn|N\sh|F\sl|M\sc|L\sv|T\ss|O\sg)'

    bracket_string = r'(?P<atom>\[.+?\])'

    subs, prod = ' '.join(subs), ' '.join(prod)
    subs = re.sub(bracket_string, space_remove, subs)
    subs = re.sub(IIIA_to_0, space_remove, subs)
    prod = re.sub(bracket_string, space_remove, prod)
    prod = re.sub(IIIA_to_0, space_remove, prod)
    return subs, prod


def smi2token(smi: str) -> str:
    pattern = r'(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])'
    token_regex = re.compile(pattern)
    tokens = [token for token in token_regex.findall(smi)]
    return " ".join(tokens)


def token2char(smi: str) -> str:
    pattern = r'(?P<atom>\[[^\]]+])'
    atom_pattern = r'(?P<atom>\[[^\]]+[A-Z][\s][a-z][^\]]+\])'
    smi = re.sub(pattern, space_add, smi)
    smi = re.sub(atom_pattern, atom_space_remove, smi)
    return smi


def token_preprocess(subs: str, prod: str) -> Tuple[str, str]:
    #subs, prod = hydro_remove(subs, prod)
    #subs, prod = atom_split(subs, prod)
    subs, prod = smi2token(subs), smi2token(prod)
    return subs, prod


def char_preprocess(subs: str, prod: str) -> Tuple[str, str]:
    subs, prod = smi2token(subs), smi2token(prod)
    subs, prod = token2char(subs), token2char(prod)
    return subs, prod

def space_split(lines: List[str]) -> List[str]:
    """
    split each symbol according to space.
    """
    return lines.split()

def token_count(lines: List[str]) -> Dict[str, int]:
    """
    collect each token and its appear count.
    """
    return collections.Counter(lines)       # list: [tuple('token', freq(token)}]

def sequence_encode(seq: List[str], code: Dict[str, int], eos_label: str) -> List[int]:
    """
    generate sequence encoding according to dict.
    """
    seq.append(eos_label)
    return [code.get(x) for x in seq]


class reaction_sequence(DataLoad):

    def __init__(self, root_dir, output_dir, data_name,
            split_type='token', load_type='csv'):

        self.root_dir = os.path.join(root_dir)
        self.output_dir = output_dir
        super(reaction_sequence, self).__init__(
            root_dir=root_dir,
            output_dir=output_dir,
            data_name=data_name,
            split_type=split_type,
            load_type=load_type,
        )
        self.symbol_dict = {}
        self.avg_len = 0
        self.reaction_count = 0
        self.max_len = 0

    def get_vocab_dict(self, extra_token=[]):
        """
        generate vocab list according to preprocess csv data.
        symbol_dict -> Dict[symbol, symbol_code]
        """
        # count frequencies of commonly used tokens
        symbol_list = []
        static_symbol_list = ['<BOS>', '<EOS>', '<PAD>', '<UNK>', '<SEP>', '<PROD>', '<SUBS>']      # , '<SEP>', '<PROD>', '<SUBS>'
        static_symbol_freq = [0 for i in range(len(static_symbol_list + extra_token))]

        for data in self.data.values():
            for subs_smi, prod_smi in zip(data['subs'], data['prod']):
                # split tokens by space' '
                subs_smi, prod_smi = subs_smi.split(), prod_smi.split()
                # different from Graph2SMILES which sets 512 in default
                self.max_len = (len(subs_smi) + len(prod_smi)) if \
                    (len(subs_smi) + len(prod_smi)) > self.max_len else self.max_len
                self.avg_len += (len(subs_smi) + len(prod_smi))
                self.reaction_count += 1
                symbol_list.extend(subs_smi)
                symbol_list.extend(prod_smi)
                # TODO: why only add 2 for '<EOS>'
                static_symbol_freq[1] += 2

        # f'token_cout' returns dict, sorted does rerank by key of dict[key]
        symbol_count = sorted(token_count(symbol_list).items(),
                              key=lambda x: x[1], reverse=True)
        symbol, symbol_freq = [x[0] for x in symbol_count], \
            [x[1] for x in symbol_count]
        symbol = static_symbol_list + extra_token + symbol
        symbol_freq = static_symbol_freq + symbol_freq
        self.vocab_freq_plot(symbol, symbol_freq)

        self.symbol_dict = {j: i for i, j in enumerate(symbol)}
        self.total_symbol_count = sum(symbol_freq)
        self.symbol_freq = [i / self.total_symbol_count for i in symbol_freq]
        self.symbol_count = symbol_freq
        self.max_len += 1
        self.avg_len += self.reaction_count
        self.avg_len = self.avg_len / self.reaction_count

    def smi_encode(self, save=True, max_len=None):
        if max_len == None:
            max_len = self.max_len
        for data, data_split in zip(self.data.values(), self.data.keys()):
            subs_enc = []
            prod_enc = []
            reaction_type = []
            reaction_len = []

            for subs_smi, prod_smi, react_type in tqdm(zip(data['subs'], data['prod'], data['reaction_type']),
                                                       desc=f'{self.data_name} is now encoding...',
                                                       total=len(data['subs'])):
                subs_smi, prod_smi = space_split(subs_smi), \
                    space_split(prod_smi)

                # transform tokens to ids
                subs_enc.append(sequence_encode(subs_smi, self.vocab_dict, '<EOS>')[:])
                prod_enc.append(sequence_encode(prod_smi, self.vocab_dict, '<EOS>')[:])
                reaction_len.append(len(subs_enc[-1]) + len(prod_enc[-1]))
                reaction_type.append(react_type)

            if save == True:
                self.encode_save(np.array(subs_enc, dtype=object), np.array(prod_enc, dtype=object),
                                 reaction_type, data_split)
                self.length_save(np.array(reaction_len), data_split)

        if save == True:
            self.dict_save()

    def encode_save(self, subs, prod, reaction_type, data_split):
        np.savez(os.path.join(self.out_dir, f'{data_split}_{self.split_type}_encoding.npz'),
                 subs=subs, prod=prod, reaction_type=reaction_type)

    def dict_save(self):
        np.savez(os.path.join(self.out_dir, f'{self.split_type}_vocab_dict.npz'), vocab_dict=self.vocab_dict,
                 seq_len=self.max_len, vocab_freq=self.symbol_freq, vocab_count=self.symbol_count)

    def length_save(self, len, data_split):
        np.savez(os.path.join(self.out_dir, f'{data_split}_{self.split_type}_length.npz'),
                 len=len)

    @property
    def vocab_dict(self) -> Dict[str, int]:
        """
        return dataset's vocab dict, which likes {symbol : symbol_code}.
        """
        return self.symbol_dict

    def vocab_freq_plot(self, symbol: List, symbol_freq: List, need_plot=False):
        """
        plot the vocab occurrence frequence.
        """
        vocab, freq = symbol, symbol_freq
        if need_plot:
            plt.bar(vocab, freq)
            plt.title('smiles symbol occurrence frequence')
            plt.xlabel('smiles symbol')
            plt.ylabel('occurrence frequence')
            plt.savefig(os.path.join(self.out_dir, 'token_count.png'))
            plt.close()

        with open(os.path.join(self.out_dir, f'{self.split_type}_token_count.txt'), 'w') as f:
            for i, j in zip(vocab, freq):
                f.writelines('{0}\t{1}\n'.format(i, j))

    @property
    def key(self) -> str:
        return self.data_name

    @property
    def in_dir(self) -> str:
        return os.path.join(self.root_dir, self.key)

    @property
    def out_dir(self) -> str:
        return os.path.join(self.output_dir, self.key)


if __name__ == '__main__':
    uspto50k_sequence = reaction_sequence('debug_uspto_50k', split_type = 'token')
    uspto50k_sequence.get_vocab_dict()
    uspto50k_sequence.smi_encode()

