from rdkit import Chem

ATOM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
             'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti',
             'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
             'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Ta', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir',
             'Ce', 'Gd', 'Ga', 'Cs', '<vnode>', '<unk>', '<seq>']
ATOM_DICT = {symbol: i for i, symbol in enumerate(ATOM_LIST)}

MAX_DEGREE = 9
DEGREE = list(range(MAX_DEGREE + 1))
HYBRIDIZATION = [Chem.rdchem.HybridizationType.SP,
                 Chem.rdchem.HybridizationType.SP2,
                 Chem.rdchem.HybridizationType.SP3,
                 Chem.rdchem.HybridizationType.SP3D,
                 Chem.rdchem.HybridizationType.SP3D2]
HYBRIDIZATION_DICT = {hb: i for i, hb in enumerate(HYBRIDIZATION)}

FORMAL_CHARGE = [-1, -2, 1, 2, 0]
FC_DICT = {fc: i for i, fc in enumerate(FORMAL_CHARGE)}

VALENCE = [0, 1, 2, 3, 4, 5, 6]
VALENCE_DICT = {vl: i for i, vl in enumerate(VALENCE)}

NUM_Hs = [0, 1, 3, 4, 5]
NUM_Hs_DICT = {nH: i for i, nH in enumerate(NUM_Hs)}

CHIRAL_TAG = [Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
              Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
              Chem.rdchem.ChiralType.CHI_UNSPECIFIED]
CHIRAL_TAG_DICT = {ct: i for i, ct in enumerate(CHIRAL_TAG)}

RS_TAG = ["R", "S", "None"]
RS_TAG_DICT = {rs: i for i, rs in enumerate(RS_TAG)}

BOND_TYPES = [None,
              Chem.rdchem.BondType.SINGLE,
              Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE,
              Chem.rdchem.BondType.AROMATIC]
BOND_FLOATS = [0.0, 1.0, 2.0, 3.0, 1.5, 0.5]  # 0.5 for virtual node
BOND_FLOATS_DICT = {f: i for i, f in enumerate(BOND_FLOATS)}
BOND_FLOAT_TO_TYPE = {
    0.0: BOND_TYPES[0],
    1.0: BOND_TYPES[1],
    2.0: BOND_TYPES[2],
    3.0: BOND_TYPES[3],
    1.5: BOND_TYPES[4],
}

BOND_STEREO = [Chem.rdchem.BondStereo.STEREOE,
               Chem.rdchem.BondStereo.STEREOZ,
               Chem.rdchem.BondStereo.STEREONONE]
BOND_STEREO_DICT = {st: i for i, st in enumerate(BOND_STEREO)}

SEQ_ATOM_FEAT_NUM = 6
MOL_ATOM_FEAT_NUM = 10
SYMBOL_SIZE = len(ATOM_LIST)
LAYER_SIZE = 7
DEGREE_SIZE = len(DEGREE)
CHARGE_SIZE = len(FORMAL_CHARGE)
VALENCY_SIZE = len(VALENCE)
RING_SIZE = 2
SUBS_PROD_SIZE = 2
HYDRO_SIZE = len(NUM_Hs)
CHIRAL_SIZE = len(CHIRAL_TAG)
RS_SIZE = len(RS_TAG)
HYBRID_SIZE = len(HYBRIDIZATION)

BOND_FEAT_NUM = 4
BOND_TYPE_SIZE = len(BOND_FLOATS) - 1
BOND_STEREO_SIZE = len(BOND_STEREO)
BOND_CONJUGATE_SIZE = 1
BOND_RING_SIZE = 1


E_CONFIG_LAYER_SIZE = 7
E_LAYER_LIST = [3, 9, 9, 19, 19, 33, 33]

NODE_FDIM = SYMBOL_SIZE + DEGREE_SIZE + CHARGE_SIZE + VALENCY_SIZE + \
    RING_SIZE + SUBS_PROD_SIZE + HYDRO_SIZE + CHIRAL_SIZE + RS_SIZE + HYBRID_SIZE
BOND_FDIM = BOND_TYPE_SIZE + BOND_STEREO_SIZE + \
    BOND_CONJUGATE_SIZE + BOND_RING_SIZE
MAX_DIST = 9
MAX_DEG = 9


def get_electronic_layer(atom_idx: int, index=True):
    layer = [0] * E_CONFIG_LAYER_SIZE
    e_layer = [2, 8, 8, 18, 18, 32, 32]

    last_e_num = 0
    for i, e_num in enumerate(e_layer):
        layer[i] = e_num if atom_idx > e_num else atom_idx
        atom_idx -= e_num
        last_e_num += e_num
        if atom_idx <= 0:
            break

    if index == False:
        layer = [i / j for i, j in zip(layer, e_layer)]

    return layer


def get_atom_feature(atom=None, atom_symbol=None, seq=False, is_ring=0, is_subs=0):
    feat_idx = []
    symbol = atom.GetSymbol() if atom != None else atom_symbol
    feat_idx.append(ATOM_DICT.get(symbol, ATOM_DICT['<unk>']))

    # len(feat_idx) = MOL_ATOM_FEAT_NUM
    if symbol in ['<vnode>', '<unk>', '<seq>']:
        feat_idx.extend([-1] * (MOL_ATOM_FEAT_NUM - 1))
    else:
        # len = SEQ_ATOM_FEAT_NUM
        feat_idx.append(atom.GetDegree() if atom.GetDegree() in DEGREE else MAX_DEGREE)
        feat_idx.append(FC_DICT.get(atom.GetFormalCharge(), 4))
        feat_idx.append(VALENCE_DICT.get(atom.GetTotalValence(), 6))
        feat_idx.append(int(atom.GetIsAromatic()) if not seq else is_ring)
        feat_idx.append(is_subs)

        if not seq:
            feat_idx.append(NUM_Hs_DICT.get(atom.GetTotalNumHs(), 4))
            feat_idx.append(CHIRAL_TAG_DICT.get(atom.GetChiralTag(), 2))
            rs_tag = atom.GetPropsAsDict().get('_CIPCode', 'None')
            feat_idx.append(RS_TAG_DICT.get(rs_tag, 2))
            feat_idx.append(HYBRIDIZATION_DICT.get(atom.GetHybridization(), 4))
        else:
            feat_idx.extend([-1] * (MOL_ATOM_FEAT_NUM - SEQ_ATOM_FEAT_NUM))

    return feat_idx


def get_bond_feature(bond=None, virtual=False):
    feat_idx = []
    if not virtual:
        assert bond != None
        feat_idx.extend([int(bond.GetBondTypeAsDouble() == i) for i in BOND_FLOATS[1:]])
        feat_idx.extend([int(bond.GetStereo() == i) for i in BOND_STEREO])
        feat_idx.append(int(bond.GetIsConjugated()))
        feat_idx.append(int(bond.IsInRing()))
    else:
        feat_idx.extend([0, 0, 0, 0, 1])
        feat_idx.extend([0] * 5)

    return feat_idx

