import sys
from pathlib import Path
_TEST_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_TEST_CODE))
sys.path.insert(0, str(_TEST_CODE / "ntdyg"))
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
NTDYG_DIR = CODE_DIR / "ntdyg"
sys.path.insert(0, str(NTDYG_DIR))

from memory import NodeMemoryModel
from model.Combine import CombinedModel
from model.CensNet import CensNet
from model.Transformer import TransformerTargetEdgeClassifier


class RecordingGraphEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 2)
        self.last_node_features = None

    def forward(self, node_features, *args, **kwargs):
        self.last_node_features = node_features.detach().clone()
        context = self.projection(node_features.mean(dim=1))
        return None, context


class RecordingFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(4, 2)
        self.last_graph_context = None

    def forward(self, raw_input, GNN_output, mask=None):
        self.last_graph_context = GNN_output.detach().clone()
        fused = torch.cat([raw_input, GNN_output], dim=-1)
        return self.classifier(fused), fused


class AblationControlTests(unittest.TestCase):
    def make_combined(self, ablation):
        config = SimpleNamespace(num_classes=2)
        graph = RecordingGraphEncoder()
        fusion = RecordingFusion()
        model = CombinedModel(config, graph, fusion, ablation=ablation, node_memory_dim=2)
        return model, graph, fusion

    def run_combined(self, model):
        node_features = torch.tensor([[[1.0, 2.0, 7.0], [3.0, 4.0, 8.0]]])
        edge_features = torch.tensor([[[0.5, -0.5]]])
        model(
            H_v=node_features,
            H_e=edge_features,
            H_e_pad=None,
            adj_e=None,
            adj_v=None,
            T=None,
            mask=torch.zeros((1, 1), dtype=torch.long),
            num_nodes=torch.tensor([2]),
            num_edges=torch.tensor([1]),
            target_edge_index=torch.tensor([0]),
        )
        return node_features

    def test_direction_agnostic_messages_are_symmetric_without_changing_parameters(self):
        kwargs = dict(
            num_nodes=2,
            feature_dim=1,
            memory_dim=3,
            message_dim=4,
            hidden_dim=5,
            classifier_hidden_dim=6,
            device="cpu",
        )
        directed = NodeMemoryModel(**kwargs)
        symmetric = NodeMemoryModel(**kwargs, direction_agnostic_messages=True)
        self.assertEqual(
            sum(p.numel() for p in directed.parameters()),
            sum(p.numel() for p in symmetric.parameters()),
        )

        src = torch.tensor([0.2, -0.3, 0.7])
        dst = torch.tensor([-0.1, 0.6, 0.4])
        event = torch.tensor([0.8])
        delta = torch.tensor([2.0])
        forward = symmetric.build_message(src, dst, event, delta)
        reverse = symmetric.build_message(dst, src, event, delta)
        self.assertTrue(torch.allclose(forward, reverse))

    def test_no_memory_guidance_zeros_only_memory_coordinates(self):
        model, graph, _ = self.make_combined("A2")
        original = self.run_combined(model)
        self.assertTrue(torch.equal(graph.last_node_features[..., :2], torch.zeros_like(graph.last_node_features[..., :2])))
        self.assertTrue(torch.equal(graph.last_node_features[..., 2:], original[..., 2:]))

    def test_no_graph_context_zeros_fusion_input(self):
        model, _, fusion = self.make_combined("A3")
        self.run_combined(model)
        self.assertTrue(torch.equal(fusion.last_graph_context, torch.zeros_like(fusion.last_graph_context)))

    def test_combined_variants_keep_the_same_parameter_count(self):
        counts = []
        for ablation in ("full", "A2", "A3"):
            model, _, _ = self.make_combined(ablation)
            counts.append(sum(p.numel() for p in model.parameters()))
        self.assertEqual(len(set(counts)), 1)

    def test_graph_ablation_variants_run_through_the_real_model(self):
        config = SimpleNamespace(
            num_classes=2,
            input_dim=4,
            n_heads=2,
            drop_out=0.0,
            n_layer=1,
        )
        for ablation in ("full", "A2", "A3"):
            model = CombinedModel(
                config,
                CensNet(3, 4, 0.0),
                TransformerTargetEdgeClassifier(config, "cpu", hidden_size=8),
                ablation=ablation,
                node_memory_dim=2,
            )
            logits, fused = model(
                H_v=torch.randn(1, 2, 3),
                H_e=torch.randn(1, 2, 4),
                H_e_pad=None,
                adj_e=torch.eye(2).unsqueeze(0),
                adj_v=torch.eye(2).unsqueeze(0),
                T=torch.ones(1, 2, 2),
                mask=torch.zeros(1, 2, dtype=torch.long),
                num_nodes=torch.tensor([2]),
                num_edges=torch.tensor([2]),
                target_edge_index=torch.tensor([0]),
            )
            self.assertEqual(tuple(logits.shape), (1, 2))
            self.assertEqual(tuple(fused.shape), (1, 8))


if __name__ == "__main__":
    unittest.main()
