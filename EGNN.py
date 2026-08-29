import os
import torch
import torch.nn as nn

EDGE_CUTOFF = float(os.environ.get("EGNN_EDGE_CUTOFF", "12.0"))


def scatter_mean(src, index, dim_size):
    out = torch.zeros(
        dim_size,
        src.size(-1),
        device=src.device,
        dtype=src.dtype,
    )

    count = torch.zeros(
        dim_size,
        1,
        device=src.device,
        dtype=src.dtype,
    )

    out.index_add_(0, index, src)

    ones = torch.ones(
        index.size(0),
        1,
        device=src.device,
        dtype=src.dtype,
    )

    count.index_add_(0, index, ones)

    return out / count.clamp(min=1.0)


def scatter_max(src, index, dim_size):
    expanded_index = index.unsqueeze(-1).expand_as(src)

    out = torch.full(
        (dim_size, src.size(-1)),
        -torch.inf,
        device=src.device,
        dtype=src.dtype,
    )

    out = out.scatter_reduce(
        0,
        expanded_index,
        src,
        reduce="amax",
        include_self=True,
    )

    return torch.where(
        torch.isfinite(out),
        out,
        torch.zeros_like(out),
    )


def scatter_attention_pool(src, logits, index, dim_size):
    logits = logits.view(-1, 1)

    expanded_index = index.unsqueeze(-1).expand_as(logits)

    max_logits = torch.full(
        (dim_size, 1),
        -torch.inf,
        device=src.device,
        dtype=src.dtype,
    )

    max_logits = max_logits.scatter_reduce(
        0,
        expanded_index,
        logits,
        reduce="amax",
        include_self=True,
    )

    stable_logits = logits - max_logits[index]
    weights = torch.exp(stable_logits)

    denom = torch.zeros(
        dim_size,
        1,
        device=src.device,
        dtype=src.dtype,
    )

    denom.index_add_(0, index, weights)

    weights = weights / denom[index].clamp(min=1e-12)

    out = torch.zeros(
        dim_size,
        src.size(-1),
        device=src.device,
        dtype=src.dtype,
    )

    out.index_add_(0, index, src * weights)

    return out


def radial_basis(dist, num_radial=16, cutoff=12.0):
    centers = torch.linspace(
        0.0,
        cutoff,
        num_radial,
        device=dist.device,
        dtype=dist.dtype,
    )

    width = cutoff / num_radial

    return torch.exp(
        -((dist - centers.view(1, -1)) ** 2)
        / (width ** 2)
    )


class EGNNLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_radial: int = 16,
        num_edge_features: int = 32,
    ):
        super().__init__()

        self.num_radial = num_radial
        self.num_edge_features = num_edge_features

        edge_in_dim = (
            2 * hidden_dim
            + num_radial
            + num_edge_features
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h, pos, edge_index, edge_attr):
        row, col = edge_index

        rel_pos = pos[row] - pos[col]
        dist2 = (rel_pos ** 2).sum(dim=-1, keepdim=True)
        dist = torch.sqrt(dist2 + 1e-8)

        radial = radial_basis(
            dist,
            num_radial=self.num_radial,
            cutoff=EDGE_CUTOFF,
        )

        edge_input = torch.cat(
            [
                h[row],
                h[col],
                radial,
                edge_attr.float(),
            ],
            dim=-1,
        )

        m_ij = self.edge_mlp(edge_input)

        m_i = scatter_mean(
            m_ij,
            row,
            dim_size=h.size(0),
        )

        h_new = h + self.node_mlp(
            torch.cat([h, m_i], dim=-1)
        )

        coord_weight = self.coord_mlp(m_ij)
        coord_update = rel_pos * coord_weight

        delta_pos = scatter_mean(
            coord_update,
            row,
            dim_size=pos.size(0),
        )

        pos_new = pos + delta_pos

        return h_new, pos_new, m_ij


class EGNNRegressor(nn.Module):
    def __init__(
        self,
        in_dim=6,
        hidden_dim=64,
        num_layers=4,
        dropout=0.1,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.layers = nn.ModuleList(
            [
                EGNNLayer(
                    hidden_dim,
                    num_radial=16,
                    num_edge_features=32,
                )
                for _ in range(num_layers)
            ]
        )

        self.edge_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.readout = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def build_chem_edge_features(
        self,
        data,
        edge_index,
        node_is_ligand,
    ):
        h = data.x.float()
        pos = data.pos.float()

        row, col = edge_index

        atom_onehot = h[:, :5].float()

        row_elem = atom_onehot[row].argmax(dim=-1)
        col_elem = atom_onehot[col].argmax(dim=-1)

        row_is_ligand = node_is_ligand[row]

        protein_elem = torch.where(
            row_is_ligand,
            col_elem,
            row_elem,
        )

        ligand_elem = torch.where(
            row_is_ligand,
            row_elem,
            col_elem,
        )

        pair_idx = protein_elem * 5 + ligand_elem

        pair_onehot = torch.zeros(
            edge_index.size(1),
            25,
            device=h.device,
            dtype=h.dtype,
        )

        pair_onehot.scatter_(
            1,
            pair_idx.view(-1, 1),
            1.0,
        )

        rel_pos = pos[row] - pos[col]
        dist = torch.sqrt(
            (rel_pos ** 2).sum(dim=-1, keepdim=True) + 1e-8
        )

        # Element order assumed from graph node features:
        # H, C, N, O, S
        vdw_radii = torch.tensor(
            [1.20, 1.70, 1.55, 1.52, 1.80],
            device=h.device,
            dtype=h.dtype,
        )

        row_vdw = vdw_radii[row_elem].view(-1, 1)
        col_vdw = vdw_radii[col_elem].view(-1, 1)
        vdw_sum = row_vdw + col_vdw

        dist_norm = (dist / EDGE_CUTOFF).clamp(min=0.0, max=1.0)
        inv_dist = 1.0 / (dist + 1e-6)

        vdw_norm = dist / vdw_sum.clamp(min=1e-6)
        vdw_overlap = torch.relu(1.0 - vdw_norm)

        # N/O/S are indices 2, 3, 4.
        row_hbond_atom = (row_elem >= 2)
        col_hbond_atom = (col_elem >= 2)

        hbond_like = (
            row_hbond_atom
            & col_hbond_atom
            & (dist.view(-1) <= 3.5)
        ).float().view(-1, 1)

        # C/S hydrophobic-ish atoms: C index 1, S index 4.
        row_hydrophobic = (row_elem == 1) | (row_elem == 4)
        col_hydrophobic = (col_elem == 1) | (col_elem == 4)

        hydrophobic_like = (
            row_hydrophobic
            & col_hydrophobic
            & (dist.view(-1) >= 3.0)
            & (dist.view(-1) <= 4.5)
        ).float().view(-1, 1)

        close_contact = (
            dist.view(-1) <= 4.0
        ).float().view(-1, 1)

        chem_scalars = torch.cat(
            [
                dist_norm,
                inv_dist,
                vdw_norm,
                vdw_overlap,
                hbond_like,
                hydrophobic_like,
                close_contact,
            ],
            dim=-1,
        )

        edge_attr = torch.cat(
            [
                pair_onehot,
                chem_scalars,
            ],
            dim=-1,
        )

        return edge_attr

    def forward(self, data):
        h = data.x.float()
        pos = data.pos.float()
        edge_index = data.edge_index

        node_is_ligand = data.x[:, 5] > 0.5

        row, col = edge_index

        cross_edge_mask = node_is_ligand[row] != node_is_ligand[col]
        edge_index = edge_index[:, cross_edge_mask]

        if edge_index.size(1) == 0:
            raise RuntimeError(
                "Graph contains no protein-ligand cross edges."
            )

        row, col = edge_index

        rel_pos = pos[row] - pos[col]
        dist = torch.sqrt((rel_pos ** 2).sum(dim=-1) + 1e-8)
        cutoff_mask = dist <= EDGE_CUTOFF
        edge_index = edge_index[:, cutoff_mask]

        if edge_index.size(1) == 0:
            raise RuntimeError(
                f"Graph contains no protein-ligand cross edges within {EDGE_CUTOFF} Å."
            )

        row, col = edge_index

        edge_attr = self.build_chem_edge_features(
            data=data,
            edge_index=edge_index,
            node_is_ligand=node_is_ligand,
        )

        if edge_attr.size(-1) != 32:
            raise RuntimeError(
                f"Expected 32 edge features, got {edge_attr.size(-1)}"
            )

        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(
                h.size(0),
                dtype=torch.long,
                device=h.device,
            )

        num_graphs = (
            int(batch.max().item()) + 1
            if batch.numel() > 0
            else 1
        )

        h = self.input_proj(h)

        last_messages = None

        for layer in self.layers:
            h, pos, last_messages = layer(
                h,
                pos,
                edge_index,
                edge_attr,
            )

        if last_messages is None:
            raise RuntimeError("No edge messages were produced.")

        edge_batch = batch[row]

        edge_logits = self.edge_gate(last_messages)

        edge_attn = scatter_attention_pool(
            last_messages,
            edge_logits,
            edge_batch,
            dim_size=num_graphs,
        )

        edge_max = scatter_max(
            last_messages,
            edge_batch,
            dim_size=num_graphs,
        )

        g = torch.cat(
            [edge_attn, edge_max],
            dim=-1,
        )

        return self.readout(g).squeeze(-1)