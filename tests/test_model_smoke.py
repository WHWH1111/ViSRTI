import torch

from models.visnet_v2 import VisNetV2


def test_visnet_v2_graph_initializes_on_cpu():
    model = VisNetV2(feature_level="graph")

    assert isinstance(model, torch.nn.Module)
    assert sum(parameter.numel() for parameter in model.parameters()) > 0
    assert next(model.parameters()).device.type == "cpu"


def test_nested_head_is_in_optimizer_without_a_warmup_forward():
    model = VisNetV2(feature_level="graph")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert all(id(parameter) in optimized for parameter in model.fusion_net.parameters())


def test_original_visnet_head_has_no_fusion_network():
    model = VisNetV2(feature_level="graph", output_head="original_visnet")

    assert model.graph_processor is None
    assert model.fusion_net is None
