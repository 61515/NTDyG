"""Numerically equivalent indexing for the DyGLib CAWN position encoder.

Only the parameter-free NumPy counts/lookups are vectorized. The inherited
learned layers and CAWN architecture are unchanged. No score or label is used.
"""
import numpy as np
import torch
from models.CAWN import PositionEncoder


class IndexedPositionEncoder(PositionEncoder):
    def count_nodes_appearances(self, src_node_ids, dst_node_ids,
                               node_interact_times, src_node_multi_hop_graphs,
                               dst_node_multi_hop_graphs):
        src_hops = [src_node_ids[:, None]] + src_node_multi_hop_graphs[0]
        dst_hops = [dst_node_ids[:, None]] + dst_node_multi_hop_graphs[0]
        self._appearance_tables = []
        for i in range(len(src_node_ids)):
            ids = np.unique(np.concatenate([h[i] for pair in zip(src_hops, dst_hops) for h in pair]))
            counts = np.zeros((len(ids), 2, self.walk_length + 1), dtype=np.float32)
            for hop, (src, dst) in enumerate(zip(src_hops, dst_hops)):
                # add.at preserves the repeated float32 additions of the native
                # loop, including repeated node appearances at the same hop.
                weight = np.float32(1 / src.shape[1])
                np.add.at(counts, (np.searchsorted(ids, src[i]), 0, hop), weight)
                np.add.at(counts, (np.searchsorted(ids, dst[i]), 1, hop), weight)
            if len(ids) and ids[0] == 0:
                counts[0] = 0
            self._appearance_tables.append((ids, counts))

    def forward(self, nodes_neighbor_ids):
        counts = np.stack([table[np.searchsorted(ids, nodes_neighbor_ids[i])]
                           for i, (ids, table) in enumerate(self._appearance_tables)])
        features = self.position_encode_layer(torch.Tensor(counts).float().to(self.device))
        return features.sum(dim=-2)
