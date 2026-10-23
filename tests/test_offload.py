"""Block swapping must be invisible to the maths: same outputs and LoRA gradients as a model that
lives entirely on the GPU, with gradient checkpointing (which re-runs the blocks in reverse).

    .venv/bin/python -m pytest tests/test_offload.py -v
"""

import copy

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from dainamo.engine.offload import (ActivationOffload, BlockSwapper, PinnedArena, make_room, park, park_optimizer,
                                    stream_layers, unpark)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d).requires_grad_(False)
        self.register_buffer("gain", torch.rand(d) + 0.5)
        self.adapter = nn.Parameter(torch.randn(d, d) * 0.01)  # trainable, like a LoRA

    def forward(self, x):
        return x + torch.tanh(self.lin(x) * self.gain + x @ self.adapter)


def run(blocks, x, ckpt):
    for b in blocks:
        x = checkpoint(b, x, use_reentrant=False) if ckpt else b(x)
    return x


@pytest.mark.parametrize("ckpt", [False, True])
def test_swapped_blocks_match_resident_blocks(ckpt):
    dev = torch.device("cuda", torch.cuda.device_count() - 1)
    torch.manual_seed(0)
    blocks = nn.ModuleList(Block(256) for _ in range(6))
    ref = copy.deepcopy(blocks).to(dev)
    swapped = copy.deepcopy(blocks)
    swapper = BlockSwapper(swapped, 4, dev, log=lambda *_: None)
    for step in range(3):
        x = torch.randn(8, 256, device=dev)
        y_ref, y = run(ref, x, ckpt), run(swapped, x, ckpt)
        assert torch.equal(y_ref, y)
        y_ref.square().mean().backward()
        y.square().mean().backward()
        for a, b in zip(ref, swapped):
            assert b.adapter.grad.device == dev and torch.equal(a.adapter.grad, b.adapter.grad)
            a.adapter.grad = b.adapter.grad = None
        # between steps the swapped blocks' frozen tensors are back in (pinned) system RAM, including
        # after checkpointing's backward recomputation (which stops before the blocks' forward hooks)
        for i in swapper.swapped:
            assert swapped[i].lin.weight.device.type == "cpu" and swapped[i].lin.weight.is_pinned()
            assert swapped[i].gain.device.type == "cpu"
        assert swapped[0].lin.weight.device == dev and not swapper.resident
    with torch.no_grad():
        x = torch.randn(8, 256, device=dev)
        assert torch.equal(run(ref, x, False), run(swapped, x, False))
    swapper.remove()


def test_every_pass_prefetches_the_block_after_the_one_running():
    """Each streamed block is copied in on the side stream while the one before it runs, in every forward pass, not
    just the first: a pass after another (which ended on the last block) or after backward (whose recomputation ran
    the blocks last to first, ending on the first streamed one) still prefetches forward."""
    dev = torch.device("cuda", torch.cuda.device_count() - 1)
    torch.manual_seed(0)
    blocks = nn.ModuleList(Block(64) for _ in range(6))
    swapper = BlockSwapper(blocks, 3, dev, log=lambda *_: None)
    copies = []  # (block, on the side stream)
    copy_in = swapper._copy_in
    swapper._copy_in = lambda i, stream: copies.append((i, stream is swapper.stream)) or copy_in(i, stream)
    x = torch.randn(8, 64, device=dev)
    with torch.no_grad():
        for _ in range(3):  # sampling steps, or a text encoder's prompts
            copies.clear()
            run(blocks, x, False)
            assert copies == [(3, True), (4, True), (5, True)]
    copies.clear()
    run(blocks, x, True).square().mean().backward()
    # forward, then the recomputation from the last block (which forward swapped out) down, fetching the one below
    # meanwhile; the last resident block's fetches the first streamed one for the next pass
    assert copies == [(3, True), (4, True), (5, True), (5, False), (4, True), (3, True), (3, True)]
    copies.clear()
    with torch.no_grad():
        run(blocks, x, False)
    assert copies == [(4, True), (5, True)]
    swapper.remove()


class TinyEncoder(nn.Module):
    """A text encoder's shape: an embedding, a vision tower's blocks and a language model's layers."""

    def __init__(self, d):
        super().__init__()
        self.embed = nn.Embedding(100, d)
        self.visual = nn.Module()
        self.visual.blocks = nn.ModuleList(Block(d) for _ in range(3))
        self.layers = nn.ModuleList(Block(d) for _ in range(5))
        self.norm = nn.LayerNorm(d)

    def forward(self, ids, image):
        for b in self.visual.blocks:
            image = b(image)
        x = self.embed(ids) + image.mean(0)
        for b in self.layers:
            x = b(x)
        return self.norm(x)


def test_streamed_text_encoder_matches_and_keeps_its_layers_in_ram():
    dev = torch.device("cuda", torch.cuda.device_count() - 1)
    torch.manual_seed(0)
    enc = TinyEncoder(256).requires_grad_(False)
    ref = copy.deepcopy(enc).to(dev)
    streamed = copy.deepcopy(enc)  # loaded to system RAM, as low VRAM mode loads it
    swappers = stream_layers(streamed, dev, log=lambda *_: None)
    assert [sw.blocks for sw in swappers] == [streamed.visual.blocks, streamed.layers]
    assert streamed.embed.weight.is_cuda and streamed.norm.weight.is_cuda

    def in_ram():
        return all(not t.is_cuda and t.is_pinned() for lst in (streamed.visual.blocks, streamed.layers)
                   for t in list(lst.parameters()) + list(lst.buffers()))
    assert in_ram()
    ids, image = torch.randint(0, 100, (12,), device=dev), torch.randn(4, 256, device=dev)
    with torch.no_grad():
        for _ in range(2):
            assert torch.equal(ref(ids, image), streamed(ids, image))
            assert in_ram() and not any(sw.resident for sw in swappers)
        # the Sandbox parks what's on the GPU between prompts; the streamed layers stay where they are
        moved = park([streamed], swappers)
        assert not streamed.embed.weight.is_cuda and in_ram()
        unpark(moved)
        assert streamed.embed.weight.is_cuda and torch.equal(ref(ids, image), streamed(ids, image))
    for sw in swappers:
        sw.remove()


def test_pinned_arena_packs_tensors_into_slabs(monkeypatch):
    monkeypatch.setattr(PinnedArena, "SLABS", (2**20,))
    arena = PinnedArena()
    torch.manual_seed(0)
    ts = [torch.randn(100, 300), torch.randn(77, 5).bfloat16(), torch.randint(-127, 127, (333,), dtype=torch.int8),
          torch.randn(3).half(), torch.empty(0, 4)]
    out = [arena.put(t) for t in ts]
    assert all(torch.equal(a, b) and b.dtype == a.dtype and b.shape == a.shape for a, b in zip(ts, out))
    assert all(b.is_pinned() and b.data_ptr() % PinnedArena.ALIGN == 0 for b in out if b.numel())
    assert len(arena.slabs) == 1  # one pinned allocation, not one per tensor
    big = arena.put(torch.ones(3 * 2**20, dtype=torch.uint8))  # bigger than a slab: its own, a power of two
    assert len(arena.slabs) == 2 and arena.slabs[1][0] == 4 * 2**20 and big.is_pinned()
    arena.put(torch.ones(1000))  # still fits in the first slab
    assert len(arena.slabs) == 2


def test_pinned_arena_picks_the_slab_size_that_wastes_least(monkeypatch):
    monkeypatch.setattr(PinnedArena, "SLABS", (2**20, 2**21, 2**22))
    mib = 2**20
    # three 1.1 MiB tensors: one each in 2 MiB slabs (smaller slabs round up to that), or all in one 4 MiB slab
    assert PinnedArena([int(1.1 * mib)] * 3).slab == 4 * mib
    # many quarter-MiB tensors fill 1 MiB slabs exactly; bigger slabs leave the last one part-empty
    assert PinnedArena([mib // 4] * 20).slab == mib
    arena = PinnedArena([int(1.1 * mib)] * 3)
    for _ in range(3):
        arena.put(torch.ones(int(1.1 * mib), dtype=torch.uint8))
    assert [s[0] for s in arena.slabs] == [4 * mib]


def test_parking_keeps_a_trained_lora_and_its_optimizer_state():
    dev = torch.device("cuda", torch.cuda.device_count() - 1)
    torch.manual_seed(0)
    ref = nn.ModuleList(Block(64) for _ in range(4)).to(dev)
    blocks = copy.deepcopy(ref)
    swapper = BlockSwapper(blocks, 2, dev, log=lambda *_: None)
    params, ref_params = [b.adapter for b in blocks], [b.adapter for b in ref]
    opt, ref_opt = torch.optim.AdamW(params, lr=1e-2), torch.optim.AdamW(ref_params, lr=1e-2)
    x = torch.randn(8, 64, device=dev)

    def step(bl, o):
        run(bl, x, False).square().mean().backward()
        o.step()
        o.zero_grad(set_to_none=True)

    step(blocks, opt)
    step(ref, ref_opt)
    # samples decode: the resident blocks, the LoRA and the optimizer state wait in system RAM
    moved = park([blocks], swapper) + park_optimizer(opt)
    assert not any(p.is_cuda for p in params) and not blocks[0].lin.weight.is_cuda
    assert not any(t.is_cuda for st in opt.state.values() for t in st.values() if torch.is_tensor(t))
    assert blocks[3].lin.weight.is_pinned()  # a streamed block's weights are left where they are
    unpark(moved)
    assert [b.adapter for b in blocks] == params and all(p.is_cuda and p.requires_grad for p in params)
    step(blocks, opt)  # training goes on as if nothing happened
    step(ref, ref_opt)
    assert all(torch.equal(a, b) for a, b in zip(params, ref_params))
    swapper.remove()


def test_make_room_parks_only_when_needed():
    dev = torch.device("cuda", torch.cuda.device_count() - 1)
    m = nn.Linear(64, 64).to(dev)
    assert make_room(0, [m], device=dev) == [] and m.weight.is_cuda
    moved = make_room(2**50, [m], device=dev)
    assert moved and not m.weight.is_cuda
    unpark(moved)
    assert m.weight.is_cuda



class SharedBlock(Block):
    def forward(self, x, shared):  # ``shared``: an input every block takes, like a timestep embedding
        return x + torch.tanh(self.lin(x) * self.gain * shared + x @ self.adapter)


def test_activation_offload_keeps_saved_inputs_in_ram_with_the_same_gradients():
    """Low VRAM training: each checkpointed block's input waits in pinned RAM from the forward pass to the
    backward pass (with blocks swapped too), with exactly the same loss and LoRA gradients. The input every
    block shares needs no grad and stays put, and each step reuses the same RAM."""
    dev = torch.device("cuda", torch.cuda.device_count() - 1)
    torch.manual_seed(0)
    blocks = nn.ModuleList(SharedBlock(256) for _ in range(6))
    ref = copy.deepcopy(blocks).to(dev)
    swapped = copy.deepcopy(blocks)
    swapper = BlockSwapper(swapped, 3, dev, log=lambda *_: None)
    with torch.cuda.device(dev):
        off = ActivationOffload(torch.device("cuda"))  # as the trainer gives it, with no index

    def loss(bl, x, shared):
        for b in bl:
            x = checkpoint(b, x, shared, use_reentrant=False)
        return x.square().mean()

    x = torch.randn(2048, 256, device=dev)  # 2 MiB per block input
    shared = torch.rand(2048, 256, device=dev) + 0.5
    held = {}
    for step in range(3):
        for _ in range(2):  # gradient accumulation: a forward and backward pass each
            ref_loss = loss(ref, x, shared)
            ref_loss.backward()
            torch.cuda.synchronize(dev)
            before = torch.cuda.memory_allocated(dev)
            with off.step():
                out = loss(swapped, x, shared)
                torch.cuda.synchronize(dev)
                held[step] = torch.cuda.memory_allocated(dev) - before
                out.backward()
            assert torch.equal(ref_loss, out)
        # blocks 1-5's inputs and the output (saved by square); not the first block's input or ``shared``
        assert off.last_bytes == 6 * x.nbytes and all(slot[2].is_pinned() for slot in off.arena.slabs)
        if step == 0:
            arena = off.arena.nbytes
        assert off.arena.nbytes == arena
        for a, b in zip(ref, swapped):
            assert torch.equal(a.adapter.grad, b.adapter.grad)
            a.adapter.grad = b.adapter.grad = None
    # the saved activations of a forward pass take one block input's worth of VRAM, not six
    assert held[2] < 2 * x.nbytes
    swapper.remove()


class TrainedBlock(nn.Module):
    """A finetune's block: every weight trains, in bf16; a buffer streams with it."""

    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)
        self.register_buffer("gain", torch.rand(d) + 0.5)

    def forward(self, x):
        return x + self.out(torch.tanh(self.lin(x) * self.gain.to(x.dtype)))


@pytest.mark.parametrize("swapped", [3, 6])  # some blocks on the GPU, or none (the first one streams too)
def test_a_finetunes_streamed_blocks_train_as_resident_ones_do(swapped):
    """Streamed blocks stepped during backward (``trainable``) end each step with exactly the weights blocks kept
    on the GPU get, the same parameters the optimizer holds, back in pinned RAM; forward passes between steps
    (samples) see them too, and ``settle`` leaves the newest weights in RAM to save. Not swapped out after a step,
    the last block backward ran is used as it is, not copied in again from RAM."""
    from dainamo.engine.finetune import FusedOptimizer

    dev = torch.device("cuda", torch.cuda.device_count() - 1)
    torch.manual_seed(0)
    blocks = nn.ModuleList(TrainedBlock(128) for _ in range(6)).to(torch.bfloat16).requires_grad_(False)
    ref = copy.deepcopy(blocks).to(dev)
    streamed = copy.deepcopy(blocks)
    swapper = BlockSwapper(streamed, swapped, dev, log=lambda *_: None, trainable=True)
    params = list(streamed.parameters())
    for m in (ref, streamed):
        m.requires_grad_(True)
    opts = []
    for m, sw in ((ref, None), (streamed, swapper)):
        opt = FusedOptimizer(m.parameters(), "adafactor", lr=1e-2, device=dev)
        if sw is not None:
            opt.before_step = sw.prepare_step
        opts.append(opt)
    for step in range(4):
        x = torch.randn(8, 128, device=dev, dtype=torch.bfloat16)
        for m in (ref, streamed):
            torch.manual_seed(step)  # the same stochastic rounding
            run(m, x, True).float().square().mean().backward()
        assert list(streamed.parameters()) == params and all(p.grad is None for p in params)
        if step % 2:
            swapper.release_all()
            for i in swapper.swapped:
                assert streamed[i].lin.weight.device.type == "cpu" and streamed[i].lin.weight.is_pinned()
            assert not swapper.resident and not swapper.dirty
        else:  # (a block on the GPU before them swapped the streamed blocks out as backward reached it)
            assert swapper.resident == swapper.dirty == ({0} if swapped == len(blocks) else set())
        with torch.no_grad():
            assert torch.equal(run(ref, x, False), run(streamed, x, False))
    swapper.settle()
    for (k, a), b in zip(ref.state_dict().items(), streamed.state_dict().values()):
        assert torch.equal(a.cpu(), b.cpu()), k
    assert not torch.equal(blocks[-1].lin.weight, streamed[-1].lin.weight)  # it trained
    swapper.remove()
