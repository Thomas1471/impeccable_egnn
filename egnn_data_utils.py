import torch


class ManualGraphBatch:
    def __init__(self, x, pos, edge_index, y, batch, num_graphs):
        self.x = x
        self.pos = pos
        self.edge_index = edge_index
        self.y = y
        self.batch = batch
        self.num_graphs = num_graphs

    def to(self, device):
        self.x = self.x.to(device)
        self.pos = self.pos.to(device)
        self.edge_index = self.edge_index.to(device)
        self.y = self.y.to(device)
        self.batch = self.batch.to(device)
        return self


def collate_graphs_plain_torch(graph_list):
    xs = []
    poss = []
    edge_indices = []
    ys = []
    batch_vecs = []

    node_offset = 0

    for graph_id, g in enumerate(graph_list):
        x = g.x
        pos = g.pos
        edge_index = g.edge_index

        n_nodes = x.size(0)

        xs.append(x)
        poss.append(pos)
        edge_indices.append(edge_index + node_offset)
        ys.append(g.y.view(-1))
        batch_vecs.append(
            torch.full((n_nodes,), graph_id, dtype=torch.long)
        )

        node_offset += n_nodes

    batch = ManualGraphBatch(
        x=torch.cat(xs, dim=0),
        pos=torch.cat(poss, dim=0),
        edge_index=torch.cat(edge_indices, dim=1),
        y=torch.cat(ys, dim=0),
        batch=torch.cat(batch_vecs, dim=0),
        num_graphs=len(graph_list),
    )

    return batch