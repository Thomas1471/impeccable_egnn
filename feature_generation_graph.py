import torch
import numpy as np

ATOM_ORDER = ["H", "C", "N", "O", "S"]
ATOM_TO_IDX = {atom: i for i, atom in enumerate(ATOM_ORDER)}


class GraphData:
    def __init__(self, x, pos, edge_index, y=None):
        self.x = x
        self.pos = pos
        self.edge_index = edge_index
        self.y = y
        self.num_graphs = 1

    def to(self, device):
        self.x = self.x.to(device)
        self.pos = self.pos.to(device)
        self.edge_index = self.edge_index.to(device)
        if self.y is not None:
            self.y = self.y.to(device)
        return self


def radius_graph_pure_torch(pos, cutoff, loop=False):
    dist = torch.cdist(pos, pos)
    mask = dist <= cutoff

    if not loop:
        mask.fill_diagonal_(False)

    edge_index = mask.nonzero(as_tuple=False).t().contiguous()
    return edge_index


# def create_pose_graph(prot_coords, lig_coords, prot_types, lig_types, cutoff=6.0):
#     pos = torch.tensor(
#         np.vstack([prot_coords, lig_coords]),
#         dtype=torch.float
#     )


    


#     all_types = prot_types + lig_types
#     is_ligand = [0] * len(prot_types) + [1] * len(lig_types)

#     node_features = []
#     for atom_type, lig_flag in zip(all_types, is_ligand):
#         one_hot = [0] * len(ATOM_ORDER)
#         one_hot[ATOM_TO_IDX[atom_type]] = 1
#         node_features.append(one_hot + [lig_flag])

#     x = torch.tensor(node_features, dtype=torch.float)
#     edge_index = radius_graph_pure_torch(pos, cutoff=cutoff, loop=False)

#     return GraphData(x=x, pos=pos, edge_index=edge_index)

def create_pose_graph(
    prot_coords,
    lig_coords,
    prot_types,
    lig_types,
    cutoff=12.0,
):
    """
    Build a protein-ligand cross-edge-only graph.

    Assumes prot_coords has already been residue-filtered upstream.

    Keeps:
      - all provided protein atoms
      - all ligand atoms
      - only protein-ligand edges within cutoff

    Does NOT create:
      - protein-protein edges
      - ligand-ligand edges
    """

    prot_coords = torch.tensor(prot_coords, dtype=torch.float)
    lig_coords = torch.tensor(lig_coords, dtype=torch.float)

    if prot_coords.numel() == 0 or lig_coords.numel() == 0:
        raise ValueError("Empty protein or ligand coordinates passed to create_pose_graph.")

    pos = torch.cat([prot_coords, lig_coords], dim=0)

    num_prot = prot_coords.size(0)
    num_lig = lig_coords.size(0)

    all_types = prot_types + lig_types
    is_ligand = [0] * len(prot_types) + [1] * len(lig_types)

    node_features = []
    for atom_type, lig_flag in zip(all_types, is_ligand):
        one_hot = [0] * len(ATOM_ORDER)
        one_hot[ATOM_TO_IDX[atom_type]] = 1
        node_features.append(one_hot + [lig_flag])

    x = torch.tensor(node_features, dtype=torch.float)

    # Protein-ligand distances only: [num_protein_atoms, num_ligand_atoms]
    dists = torch.cdist(prot_coords, lig_coords)

    prot_idx, lig_idx = torch.where(dists <= cutoff)

    if prot_idx.numel() == 0:
        raise ValueError("No protein-ligand edges found within cutoff.")

    # Ligand indices are offset because ligand nodes come after protein nodes.
    lig_idx = lig_idx + num_prot

    # Add both directions for message passing.
    edge_prot_to_lig = torch.stack([prot_idx, lig_idx], dim=0)
    edge_lig_to_prot = torch.stack([lig_idx, prot_idx], dim=0)

    edge_index = torch.cat(
        [edge_prot_to_lig, edge_lig_to_prot],
        dim=1,
    ).long().contiguous()

    return GraphData(x=x, pos=pos, edge_index=edge_index)