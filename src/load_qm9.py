from rdkit import Chem
from rdkit.Chem import SDMolSupplier
import torch
from torch_geometric.data import Data
import pandas as pd
from tqdm import tqdm

HOMO_MEAN = 0.011124
HOMO_STD = 0.046936

def mol_to_graph_obj(mol, target):
    """
    Converts an RDKit mol object and a target value into a PyTorch Geometric Data object.
    """

    # --- Node feature ---
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),        # Atomic number
            int(atom.GetIsAromatic()),  # Aromaticity
            atom.GetDegree(),           # Number of bonds
            atom.GetImplicitValence(),  # Valence
            atom.GetFormalCharge(),     # Charge
        ]
        atom_features.append(features)
    
    x = torch.tensor(atom_features, dtype=torch.float)

    # --- Edge list and features ---
    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # Undirected edges (i→j and j→i)
        edge_index += [[i,j],[j,i]]

        bond_features = [
            bond.GetBondTypeAsDouble(),     # 1.0, 2.0, etc.
            int(bond.GetIsConjugated()),
            int(bond.IsInRing())
        ]

        edge_attr += [bond_features, bond_features] # Add for both directions

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # --- Target value ---
    target = float(target)
    y = torch.tensor([(target - HOMO_MEAN) / HOMO_STD], dtype=torch.float) # normalization

    # --- create PyG Data object ---
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    return data



def get_graph_data_list():
    sdf_path = "../data/QM9/raw/gdb9.sdf"
    csv_path =  "../data/QM9/raw/gdb9.sdf.csv"

    # Load molecules
    supplier = SDMolSupplier(sdf_path, removeHs=False, sanitize=False)

    # Filter out None and sanitize properly
    mols = []
    for mol in supplier:
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
            mols.append(mol)
        except:
            continue

    # Load target values
    df = pd.read_csv(csv_path)
    target_index = 6  # HOMO
    targets = df.iloc[:, target_index].values

    # Align number of targets to sanitized molecules
    assert len(mols) <= len(targets)
    
    # Convert to Data objects
    graph_data_list = []
    for i in tqdm(range(len(mols))):
        try:
            data = mol_to_graph_obj(mols[i], targets[i])
            graph_data_list.append(data)
        except:
            continue

    return graph_data_list

if __name__ == "__main__":
    data_list = get_graph_data_list()
    print(f"Prepared {len(data_list)} graphs.")

