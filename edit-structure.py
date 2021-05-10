import numpy as np
from operator import itemgetter

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS
from rdkit.Chem.rdchem import EditableMol
from random import randint
from rdkit.Chem.AllChem import EmbedMolecule
from rdkit.Geometry import Point3D


def replace(mol, p, r, addHs=True, chiral=False):

    p = Chem.MolFromSmarts(p) if isinstance(p, str) else p
    r = Chem.MolFromSmarts(r) if isinstance(r, str) else r
    ref = Chem.Mol(mol)
    del mol

    mols = Chem.ReplaceSubstructs(ref, p, r, useChirality=chiral)
    mols = list(mols)

    for i, mol in enumerate(mols):
        if chiral:
            Chem.AssignAtomChiralTagsFromStructure(mol)
            AllChem.EmbedMolecule(mol)

        AllChem.SanitizeMol(mol)

        if addHs:
            mols[i] = mol = Chem.AddHs(mol)

        mols[i] = align(mol, ref=ref)

    return mols


def get_name(atom):
    if mi := atom.GetMonomerInfo():
        return mi.GetName().strip()
    return None


def match_core(mol, *, ref):

    ref = Chem.Mol(ref)
    mol = Chem.Mol(mol)

    common = rdFMCS.FindMCS(
        [ref, mol],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        matchChiralTag=True,
    )
    common = Chem.MolFromSmarts(common.smartsString)
    matches = ref.GetSubstructMatches(common)
    ref_conf = ref.GetConformer()

    for match in matches:
        mached_atom = dict()

        for idx in match:
            ref_at = ref.GetAtomWithIdx(idx)
            ref_at_name = get_name(ref_at)
            ref_at_pos = ref_conf.GetPositions()[idx]
            mached_atom[ref_at_name] = ref_at_pos

        new = Chem.Mol(ref)
        eref = Chem.EditableMol(new)

        for at in ref.GetAtoms():
            idx = at.GetIdx()
            if idx not in match:
                eref.ReplaceAtom(idx, Chem.Atom('*'))

        core = eref.GetMol()
        core = AllChem.DeleteSubstructs(core, Chem.MolFromSmiles('*'))

        yield match, core


def get_named_coords(mol):

    mol_at_with_name = dict()
    mol_conf = mol.GetConformer().GetPositions()

    for i, mol_at in enumerate(mol.GetAtoms()):
        if not (mi := mol_at.GetMonomerInfo()):
            continue
        mol_at_name = mi.GetName().strip()
        mol_at_pos = mol_conf[i]
        mol_at_with_name[mol_at_name] = mol_at_pos

    return mol_at_with_name


def compare_named_coords(arg: dict, *, ref: dict):
    error = 0
    for name, coord in arg.items():
        ref_coord = ref.get(name, None)
        if ref_coord is None:
            continue
        error += np.linalg.norm(coord - ref_coord)
    return error


def compare_with_names(mol, *, ref):
    mol_named_coord = get_named_coords(mol)
    ref_named_coord = get_named_coords(ref)
    return compare_named_coords(mol_named_coord, ref=ref_named_coord)


def restore_coords(mol, *, ref):

    mol = Chem.Mol(mol)
    conf = mol.GetConformer()
    ref_d = get_named_coords(ref)

    for idx, at in enumerate(mol.GetAtoms()):
        if not (mi := at.GetMonomerInfo()):
            continue
        name = mi.GetName().strip()
        if name in ref_d:
            coords = ref_d[name]
            conf.SetAtomPosition(idx, Point3D(*coords))

    return mol


def align(mol, *, ref):

    ref = Chem.Mol(ref)
    align_score = dict()

    for _, core in match_core(mol, ref=ref):
        new = Chem.Mol(mol)

        for seed in range(100):
            if AllChem.ConstrainedEmbed(new, core, randomseed=seed):  # is problem here?
                break
        else:
            raise Exception('I can\'t')

        align_score[new] = compare_with_names(new, ref=ref)

    _ = min(align_score, key=align_score.get)
    return restore_coords(_, ref=ref)
