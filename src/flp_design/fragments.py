from itertools import combinations_with_replacement

import pandas as pd
from rdkit import Chem


P_SUBSTITUENTS = {
    'phenyl': 'c1ccccc1',
    'mesityl': 'c1c(C)cc(C)cc1C',
    'tert_butyl': 'C(C)(C)C',
    'cyclohexyl': 'C1CCCCC1',
    'isopropyl': 'C(C)C',
}

N_SUBSTITUENTS = {
    'methyl': 'C',
    'ethyl': 'CC',
    'isopropyl': 'C(C)C',
    'cyclohexyl': 'C1CCCCC1',
    'phenyl': 'c1ccccc1',
}

B_ARYLS = {
    'phenyl': 'c1ccccc1',
    'p_tolyl': 'c1ccc(C)cc1',
    'mesityl': 'c1c(C)cc(C)cc1C',
    'pentafluorophenyl': 'c1c(F)c(F)c(F)c(F)c1F',
    'pentachlorophenyl': 'c1c(Cl)c(Cl)c(Cl)c(Cl)c1Cl',
    'bis_cf3_phenyl': 'c1cc(C(F)(F)F)cc(C(F)(F)F)c1',
}

LINKERS = {f'C{length}': 'C' * length for length in range(1, 7)}


def assemble_flp(lb_atom, lb_1, lb_2, linker, la_1, la_2):
    smiles = f'{lb_atom}({lb_1})({lb_2}){linker}B({la_1})({la_2})'
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def enumerate_fragment_library():
    rows = []

    for lb_atom, substituents in [('P', P_SUBSTITUENTS), ('N', N_SUBSTITUENTS)]:
        for lb_1, lb_2 in combinations_with_replacement(substituents.items(), 2):
            for la_1, la_2 in combinations_with_replacement(B_ARYLS.items(), 2):
                for linker_name, linker in LINKERS.items():
                    smiles = assemble_flp(
                        lb_atom,
                        lb_1[1],
                        lb_2[1],
                        linker,
                        la_1[1],
                        la_2[1],
                    )
                    rows.append({
                        'lb_atom': lb_atom,
                        'lb_substituent_1': lb_1[0],
                        'lb_substituent_2': lb_2[0],
                        'linker': linker_name,
                        'la_substituent_1': la_1[0],
                        'la_substituent_2': la_2[0],
                        'fragment_tokens': (
                            f'<LB_{lb_atom}><LB1_{lb_1[0]}><LB2_{lb_2[0]}>'
                            f'<LINKER_{linker_name}><LA1_{la_1[0]}><LA2_{la_2[0]}>'
                        ),
                        'canonical_smiles': smiles,
                    })

    return pd.DataFrame(rows).drop_duplicates('canonical_smiles').reset_index(drop=True)

