"""Saved LoRAs leave out the layers whose adapter is zero (``LoRANetwork.save``): the file must change the model
exactly as the whole one does, in Dainamo (``load``) and in a loader that applies what a file has (the Sandbox's
``AdapterStack``), and a layer left out must load as zero.

    .venv/bin/python -m pytest tests/test_lora_save.py -v
"""

import pytest
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from dainamo.engine.lora import AdapterStack, LoRANetwork


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)  # the same base weights in every copy
        self.blocks = nn.ModuleList([nn.Linear(48, 96), nn.Linear(96, 48), nn.Linear(48, 48)])

    def forward(self, x):
        return self.blocks[2](self.blocks[1](torch.relu(self.blocks[0](x))))


def network(kind="lora", trained=("blocks.0", "blocks.2")):
    """A network on ``Net`` whose ``trained`` layers have changes; the others are as they start (B or w2 zero)."""
    net = LoRANetwork(Net().double(), ["blocks.*"], 4, 8, kind=kind, factor=4)
    net.to("cpu", torch.float64)
    torch.manual_seed(0)
    with torch.no_grad():
        for name in trained:
            m = net.modules[name]
            (m.lokr_w2 if kind == "lokr" else m.lora_B.weight).normal_()
    return net


def layers(sd):
    """The layers a saved state dict has ("diffusion_model.blocks.0.lora_A.weight" -> "blocks.0")."""
    return {".".join(k.split(".")[1:3]) for k in sd}


@pytest.mark.parametrize("kind", ["lora", "lokr"])
def test_a_layer_with_no_change_is_left_out_and_the_file_changes_the_model_the_same(tmp_path, kind):
    net = network(kind)
    path, whole = str(tmp_path / "l.safetensors"), str(tmp_path / "whole.safetensors")
    net.save(path, dtype=torch.float64)
    save_file(net.state_dict(torch.float64), whole)
    assert layers(load_file(path)) == {"blocks.0", "blocks.2"}
    assert layers(load_file(whole)) == {"blocks.0", "blocks.1", "blocks.2"}
    x = torch.randn(3, 48, dtype=torch.float64)

    # Dainamo, loading it into a network whose blocks.1 has changes of its own (the strength sweep's case)
    other = network(kind, trained=("blocks.0", "blocks.1", "blocks.2"))
    other.load(path)
    assert torch.equal(other.model(x), net.model(x))
    adapter = other.modules["blocks.1"]
    assert not (adapter.lokr_w2 if kind == "lokr" else adapter.lora_B.weight).any()

    # a loader that applies only the layers a file has, against the whole file
    outs = []
    for f in (path, whole):
        model = Net().double()
        AdapterStack(model, "cpu", torch.float64).set([("l", load_file(f), 1.0)])
        outs.append(model(x))
    assert torch.equal(outs[0], outs[1])
    assert not torch.equal(outs[0], Net().double()(x))  # and it does change the model


def test_what_counts_is_the_adapter_as_saved():
    net = network(trained=("blocks.0",))
    with torch.no_grad():
        net.modules["blocks.1"].lora_B.weight[0, 0] = 1e-42  # below bfloat16's smallest: saved as zero there
    assert layers(net.state_dict(torch.float32, changed_only=True)) == {"blocks.0", "blocks.1"}
    assert layers(net.state_dict(torch.bfloat16, changed_only=True)) == {"blocks.0"}
    with torch.no_grad():
        net.modules["blocks.2"].lora_A.weight.zero_()  # A zero and B not is no change either
        net.modules["blocks.2"].lora_B.weight.normal_()
        net.modules["blocks.1"].lora_B.weight[0, 0] = float("nan")  # a NaN is kept, not hidden
    assert layers(net.state_dict(torch.float32, changed_only=True)) == {"blocks.0", "blocks.1"}


def test_the_resume_state_keeps_every_layer():
    assert layers(network(trained=("blocks.0",)).state_dict(torch.float32)) == {"blocks.0", "blocks.1", "blocks.2"}


@pytest.mark.parametrize("kind", ["lora", "lokr"])
def test_a_lora_with_no_changes_at_all_is_saved_whole(tmp_path, kind):
    net = network(kind, trained=())
    path = str(tmp_path / "l.safetensors")
    net.save(path)
    assert layers(load_file(path)) == {"blocks.0", "blocks.1", "blocks.2"}
    other = network(kind)
    other.load(path)
    x = torch.randn(3, 48, dtype=torch.float64)
    assert torch.equal(other.model(x), Net().double()(x))


def test_a_file_without_this_networks_layers_or_with_part_of_one_raises(tmp_path):
    net = network()
    with pytest.raises(ValueError, match="none of this network's 3 layers"):
        net.load_state_dict({"diffusion_model.other.lora_A.weight": torch.zeros(4, 48)})
    sd = net.state_dict(torch.float64)
    del sd["diffusion_model.blocks.1.lora_B.weight"]
    with pytest.raises(KeyError):
        net.load_state_dict(sd)
