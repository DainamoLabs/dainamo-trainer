"""Block swapping: run a transformer whose blocks don't all fit in VRAM.

The first blocks stay on the GPU; the rest keep their frozen tensors in pinned system RAM and are
copied in just before they run. While one block computes, the next one is copied on a side stream.
The direction is inferred from the call order, so gradient-checkpoint recomputation (which runs the
blocks last to first during backward) prefetches correctly too.

Only frozen tensors are swapped (base weights, quantization scales, norms). Trainable parameters,
i.e. LoRA weights, stay on the GPU with their optimizer state. When a block is swapped out, its
module attributes are pointed at the CPU copies with *new* tensor objects; GPU tensors that autograd
saved for backward stay alive exactly as long as autograd needs them.

A finetune's blocks (``trainable``) are swapped whole, weights being trained included. Those are swapped
through ``.data``, so the parameters the optimizer holds stay the ones the blocks use. Their optimizer steps
during backward (``finetune.FusedOptimizer``), which tells the swapper first (``prepare_step``): the block is
normally on the GPU for its recomputation by then, but a gradient that needs nothing saved (a bias added last)
can arrive before that, and the block is then brought in early. A block that was stepped is copied back to RAM
on the side stream when it's swapped out, before its next copy in reads it. Gradient checkpointing is
required: its recomputation is what brings a block back for backward, and it saves a trained weight as a new
tensor (``detach``), which pointing the parameter back at RAM leaves alone; without it, autograd would save
the parameter itself.

A block is normally swapped out by its forward hook, but non-reentrant gradient checkpointing stops
the backward recomputation as soon as it has what it needs, so that hook never fires. Every block's
pre-hook therefore also swaps out any other block still resident. Training needs gradient
checkpointing for swapping to save memory at all: without it, every block's weights stay saved for
backward.

``stream_layers`` does the same for a model that only runs forward (a text encoder in low VRAM mode):
every one of its layers is streamed, and so is each large part outside them (an embedding table, a big
projection), so only the rest of it stays on the GPU. ``ActivationOffload`` keeps training's saved activations in
pinned RAM from the forward pass to the backward pass, in low VRAM mode too.

Streamed tensors are packed into pinned slabs (``PinnedArena``): PyTorch rounds each pinned allocation
up to a power of two, which pinned one by one would cost MiniMax H3's blocks 1.6x their size in RAM, and
the slab size that packs them best varies by model (256 MiB slabs would still waste 3.1 GB on H3's
blocks, 1 GiB slabs 0.9 GB; LTX 2.5's pack best into 256 MiB).
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn


def _frozen_tensors(block: nn.Module, trainable: bool = False):
    """(module, kind, name, tensor) for every frozen parameter and buffer in the block; ``trainable``: for
    every parameter, kind "data" (swapped through ``.data``), and buffer."""
    for module in block.modules():
        for name, p in list(module._parameters.items()):
            if p is not None and (trainable or not p.requires_grad):
                yield module, "data" if trainable else "param", name, p
        for name, b in list(module._buffers.items()):
            if b is not None:
                yield module, "buffer", name, b


def _assign(module: nn.Module, kind: str, name: str, tensor: torch.Tensor):
    if kind == "param":
        module._parameters[name] = nn.Parameter(tensor, requires_grad=False)
    elif kind == "data":
        module._parameters[name].data = tensor
    else:
        module._buffers[name] = tensor


def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


class PinnedArena:
    """Pinned system RAM handed out from slabs whose sizes are powers of two, so PyTorch's rounding of each
    pinned allocation costs nothing. Given the sizes of the tensors it will hold (in the order they'll be
    put), it picks the slab size that wastes least on them; a tensor bigger than a slab gets one of the next
    power of two, which later tensors share. Without CUDA, tensors are plain CPU copies."""

    SLABS = (2**28, 2**29, 2**30, 2**31)  # 256 MiB to 2 GiB
    ALIGN = 512  # bytes; keeps every tensor's offset a multiple of its element size

    def __init__(self, sizes: list[int] = ()):
        sizes = [n for n in sizes if n]
        self.slab = min(self.SLABS, key=lambda size: self._waste(sizes, size)) if sizes else self.SLABS[0]
        self.slabs: list[list] = []  # [bytes, bytes used, the pinned slab]

    @classmethod
    def _take(cls, slabs: list[list], n: int, slab: int) -> tuple[list, int]:
        """(slab, offset) for ``n`` bytes: the first of ``slabs`` with room, or a new one (not allocated yet)."""
        slot = next((s for s in slabs if s[1] + n <= s[0]), None)
        if slot is None:
            slot = [max(slab, 1 << (n - 1).bit_length()), 0, None]
            slabs.append(slot)
        offset, slot[1] = slot[1], -(-(slot[1] + n) // cls.ALIGN) * cls.ALIGN
        return slot, offset

    @classmethod
    def _waste(cls, sizes: list[int], slab: int) -> int:
        slabs = []
        for n in sizes:
            cls._take(slabs, n, slab)
        return sum(s[0] for s in slabs) - sum(sizes)

    def empty_like(self, t: torch.Tensor) -> torch.Tensor:
        """Pinned room for a contiguous copy of ``t`` (needs CUDA)."""
        n = _nbytes(t)
        if n == 0:
            return torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
        slot, offset = self._take(self.slabs, n, self.slab)
        if slot[2] is None:
            slot[2] = torch.empty(slot[0], dtype=torch.uint8, pin_memory=True)
        return slot[2][offset:offset + n].view(t.dtype).view(t.shape)

    def put(self, t: torch.Tensor) -> torch.Tensor:
        t = t.detach()
        if not torch.cuda.is_available():
            return t.cpu()
        out = self.empty_like(t)
        out.copy_(t)
        return out

    def reset(self):
        """Hand the slabs out again from the start: nothing put so far is used any more."""
        for slot in self.slabs:
            slot[1] = 0

    @property
    def nbytes(self) -> int:
        return sum(slot[0] for slot in self.slabs if slot[2] is not None)


class BlockSwapper:
    def __init__(self, blocks: nn.ModuleList, num_swapped: int, device: torch.device, log=print,
                 arena: PinnedArena | None = None, trainable: bool = False):
        self.blocks = blocks
        self.device = device
        self.trainable = trainable
        n = len(blocks)
        num_swapped = max(0, min(int(num_swapped), n))
        self.swapped = list(range(n - num_swapped, n))
        self.cpu: dict[int, list] = {}
        self.stream = torch.cuda.Stream(device=device) if torch.cuda.is_available() else None
        self.prefetched: dict[int, tuple] = {}  # i -> (event, [gpu tensors])
        self.resident: set[int] = set()  # swapped blocks currently pointed at GPU copies
        self.last = -1  # the streamed block that ran last
        self.direction = 1  # the way the blocks ran then: 1 forward, -1 backward's recomputation
        self.handles = []
        self.owner: dict[int, int] = {}  # a streamed block's parameter (id) -> the block's index
        self.dirty: set[int] = set()  # streamed blocks stepped since they were copied in (a finetune's)
        frozen = {i: list(_frozen_tensors(blocks[i], trainable)) for i in self.swapped}
        if arena is None:  # (one shared with other swappers was planned for all of their tensors)
            arena = PinnedArena([_nbytes(t) for i in self.swapped for *_, t in frozen[i]])
        for i, block in enumerate(blocks):
            self.handles.append(block.register_forward_pre_hook(self._make_release(i)))
            if i in self.swapped:
                entries = []
                for module, kind, name, t in frozen.pop(i):
                    cpu = arena.put(t)
                    _assign(module, kind, name, cpu)
                    entries.append((module, kind, name, cpu))
                self.cpu[i] = entries
                self.owner.update((id(p), i) for p in block.parameters())
                # trainable tensors (LoRA) live on the GPU regardless
                for p in block.parameters():
                    if p.requires_grad and not trainable:
                        p.data = p.data.to(device)
                self.handles.append(block.register_forward_pre_hook(self._make_pre(i)))
                self.handles.append(block.register_forward_hook(self._make_post(i)))
            else:
                block.to(device)
        if num_swapped:
            gb = sum(_nbytes(t) for e in self.cpu.values() for *_, t in e) / 1e9
            log(f"  block swap: {n - num_swapped} of {n} blocks on the GPU, {num_swapped} streamed from RAM ({gb:.1f} GB)"
                + (", each copied back after its update" if trainable else ""))

    def _copy_in(self, i: int, stream) -> list:
        # The copies are allocated on the compute stream and only filled on the copy stream. Memory
        # the copy stream allocated (or that was recorded as in use there) would only become free
        # again once that always-busy stream caught up: several GB of reserved-but-idle VRAM with
        # large blocks. The compute stream waits for the copy before using a block, so a copy freed
        # after use can safely go straight back to the compute stream's cache.
        dst = [torch.empty_like(t, device=self.device) for *_, t in self.cpu[i]]
        with torch.cuda.stream(stream):
            for d, (*_, t) in zip(dst, self.cpu[i]):
                d.copy_(t, non_blocking=True)
        return dst

    def _prefetch(self, i: int):
        if i not in self.cpu or i in self.prefetched or self.stream is None:
            return
        # the side stream must not read pinned memory before earlier work that used it is queued
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        tensors = self._copy_in(i, self.stream)
        event = torch.cuda.Event()
        event.record(self.stream)
        self.prefetched[i] = (event, tensors)

    def _release(self, i: int):
        if i in self.dirty:
            self._write_back(i)
        for mod, kind, name, cpu in self.cpu[i]:
            _assign(mod, kind, name, cpu)
        self.resident.discard(i)

    def _write_back(self, i: int):
        """Copy a stepped block's weights back to RAM, on the side stream once the steps are done, so the copy in
        that reads them next is queued after it. Each GPU copy's memory is kept until it's been read."""
        self.dirty.discard(i)
        if self.stream is None:
            for mod, kind, name, cpu in self.cpu[i]:
                if kind == "data":
                    cpu.copy_(mod._parameters[name].data)
            return
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.stream):
            for mod, kind, name, cpu in self.cpu[i]:
                if kind == "data":
                    gpu = mod._parameters[name].data
                    cpu.copy_(gpu, non_blocking=True)
                    gpu.record_stream(self.stream)

    def prepare_step(self, p: torch.Tensor):
        """A finetune's optimizer is about to step ``p``: if it's a streamed block's, the block is brought to the GPU
        (if a gradient came before its recomputation did), and its weights are copied back to RAM when it's
        swapped out."""
        i = self.owner.get(id(p))
        if i is not None:
            self._bring_in(i)
            self.dirty.add(i)

    def _make_release(self, i: int):
        first = self.swapped[0] if self.swapped else None

        def hook(module, args):
            for j in list(self.resident):
                if j != i:
                    self._release(j)
            if first is not None and i == first - 1:
                self._prefetch(first)  # the last resident block runs while the first streamed one copies
        return hook

    def _bring_in(self, i: int):
        """Point block ``i`` at GPU copies: the one prefetched, or one made now. A block still on the GPU (left there
        by a recomputation that stopped early, or brought in for a step) is used as it is: a finetune's copy there
        may be newer than the one in RAM."""
        if i in self.resident:
            return
        current = torch.cuda.current_stream(self.device)
        if i in self.prefetched:
            event, tensors = self.prefetched.pop(i)
            current.wait_event(event)
        else:
            tensors = self._copy_in(i, current)
        for (mod, kind, name, _), t in zip(self.cpu[i], tensors):
            _assign(mod, kind, name, t)
        self.resident.add(i)

    def _make_pre(self, i: int):
        def hook(module, args):
            self._bring_in(i)
            # backward's recomputation starts on the block forward ended with, run again, then goes down a block at a
            # time. Anything else is a forward pass, such as the next one starting on the first streamed block after
            # a forward pass (which ended on the last block) or after backward (which ended on this one).
            backward = i == self.last if self.direction > 0 else i == self.last - 1
            self.direction = -1 if backward else 1
            self.last = i
            self._prefetch(i + self.direction)
        return hook

    def _make_post(self, i: int):
        def hook(module, args, output):
            self._release(i)
        return hook

    def release_all(self):
        """Swap out whatever a stopped recomputation left on the GPU (e.g. before sampling)."""
        for j in list(self.resident):
            self._release(j)

    def settle(self):
        """Swap everything out and wait for the copies back to RAM: the streamed blocks' weights there are then
        up to date (to save them)."""
        self.release_all()
        if self.stream is not None:
            self.stream.synchronize()

    def drop_prefetched(self):
        """Free copies made ahead of use (the copy stream finishes with them first)."""
        if self.stream is not None and self.prefetched:
            self.stream.synchronize()
        self.prefetched.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.drop_prefetched()


# a module outside the layer lists with this much of its own weight (an embedding table, LTX's text
# projections) is streamed on its own in low VRAM mode
BIG_MODULE = 128 * 2**20


def _own_bytes(module: nn.Module) -> int:
    return sum(_nbytes(t) for t in list(module._parameters.values()) + list(module._buffers.values()) if t is not None)


def stream_layers(model: nn.Module, device: torch.device, log=print) -> list[BlockSwapper]:
    """Run a model loaded to system RAM with only one or two layers on the GPU at a time: each list of
    repeated layers (a ModuleList named ``layers`` or ``blocks``, e.g. a language model's and its vision
    tower's) is streamed in as it runs, each module of ``BIG_MODULE`` or more outside them is copied in just
    for its own forward, and everything else moves to ``device``. Returns the swappers."""
    lists = [m for name, m in model.named_modules()
             if isinstance(m, nn.ModuleList) and name.rsplit(".", 1)[-1] in ("layers", "blocks") and len(m) > 1]
    inside = {id(sub) for lst in lists for sub in lst.modules()}
    big = [m for m in model.modules() if id(m) not in inside and _own_bytes(m) >= BIG_MODULE]
    inside.update(id(m) for m in big)
    for module in model.modules():
        if id(module) in inside:
            continue
        for kind, store in (("param", module._parameters), ("buffer", module._buffers)):
            for name, t in list(store.items()):
                if t is not None and t.device.type != torch.device(device).type:
                    _assign(module, kind, name, t.detach().to(device))
    # one arena for all of them, the large parts first so the room left in their slabs takes layers
    order = [nn.ModuleList([m]) for m in big] + lists
    arena = PinnedArena([_nbytes(t) for lst in order for block in lst for *_, t in _frozen_tensors(block)])
    made = [BlockSwapper(lst, len(lst), device, log=lambda *_: None, arena=arena) for lst in order]
    swappers = made[len(big):] + made[:len(big)]
    ram = sum(_nbytes(t) for sw in swappers for e in sw.cpu.values() for *_, t in e)
    gpu = sum(_nbytes(t) for t in list(model.parameters()) + list(model.buffers()) if t.is_cuda)
    log(f"  low VRAM: {sum(len(lst) for lst in lists)} layers"
        + (f" and {len(big)} large part{'s' if len(big) > 1 else ''}" if big else "")
        + f" streamed from RAM ({ram / 1e9:.1f} GB), {gpu / 1e9:.1f} GB on the GPU")
    return swappers


class _Saved:
    """A saved activation in pinned RAM: ``out`` records when its copy there is done; ``gpu`` is a copy
    fetched ahead of use, ready at ``ready``."""
    __slots__ = ("cpu", "out", "index", "gpu", "ready", "used")

    def __init__(self, cpu: torch.Tensor, out, index: int):
        self.cpu, self.out, self.index = cpu, out, index
        self.gpu = self.ready = None
        self.used = False


class ActivationOffload:
    """Training's saved activations in pinned system RAM (low VRAM mode). With gradient checkpointing,
    autograd keeps each block's input until backward reruns the block, which adds up to a block's input per
    block. Inside ``step()``, every saved tensor a LoRA's gradient flows through (one that requires grad and
    has a grad_fn: each block's input, after the first block) is copied to RAM on a side stream as it's
    saved, freeing the GPU copy, and copied back just before backward needs it, the one before it fetched
    while a block's backward runs. Frozen weights and the inputs every block shares (timestep embeddings,
    rotary tables, masks) don't need grad and stay where they are, as do tensors under ``MIN_BYTES``.

    The pinned slabs (``PinnedArena``) are reused every step, so they hold the largest step's activations."""

    MIN_BYTES = 2**20

    def __init__(self, device: torch.device):
        self.device = torch.device(device)
        if self.device.index is None:  # "cuda", as the trainer has it: the tensors say "cuda:0"
            self.device = torch.device(self.device.type, torch.cuda.current_device())
        self.stream = torch.cuda.Stream(device=self.device)
        self.arena = PinnedArena()
        self.largest = 0  # the largest tensor so far, which sets the slab size
        self.saved: list[_Saved] = []
        self.last_bytes = 0  # what the last step offloaded

    @contextlib.contextmanager
    def step(self):
        """Offload what a forward pass run inside saves; its backward pass has to run inside too, since the
        next step reuses the RAM."""
        self.arena.reset()
        self.saved = []
        try:
            with torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack):
                yield
        finally:
            if any(h.gpu is not None for h in self.saved):
                self.stream.synchronize()  # copies fetched but not used: let them land before they're freed
            self.last_bytes = sum(_nbytes(h.cpu) for h in self.saved)
            self.saved = []

    def _pack(self, t: torch.Tensor):
        if not (t.is_cuda and t.requires_grad and t.grad_fn is not None and t.is_contiguous()) \
                or t.device != self.device or _nbytes(t) < self.MIN_BYTES:
            return t
        n = _nbytes(t)
        if n > self.largest:  # slabs of at least 8 tensors, so their unused ends waste at most 1/8
            self.largest = n
            self.arena.slab = next((s for s in PinnedArena.SLABS if s >= 8 * n), PinnedArena.SLABS[-1])
        cpu = self.arena.empty_like(t)
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.stream):
            cpu.copy_(t, non_blocking=True)
        t.record_stream(self.stream)  # its memory isn't reused until the copy is done
        out = torch.cuda.Event()
        out.record(self.stream)
        h = _Saved(cpu, out, len(self.saved))
        self.saved.append(h)
        return h

    def _fetch(self, h: _Saved, stream) -> torch.Tensor:
        # allocated on the compute stream, as BlockSwapper._copy_in explains; the copy waits for that stream
        # (which may still be using the memory) and for the copy out
        gpu = torch.empty_like(h.cpu, device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        stream.wait_event(h.out)
        with torch.cuda.stream(stream):
            gpu.copy_(h.cpu, non_blocking=True)
        return gpu

    def _unpack(self, h):
        if not isinstance(h, _Saved):
            return h
        current = torch.cuda.current_stream(self.device)
        if h.gpu is not None:
            current.wait_event(h.ready)
            gpu, h.gpu = h.gpu, None
        else:
            gpu = self._fetch(h, current)
        h.used = True
        # backward runs the blocks last to first: fetch the activation saved before this one meanwhile
        prev = self.saved[h.index - 1] if 0 < h.index < len(self.saved) and self.saved[h.index] is h else None
        if prev is not None and not prev.used and prev.gpu is None:
            prev.gpu = self._fetch(prev, self.stream)
            prev.ready = torch.cuda.Event()
            prev.ready.record(self.stream)
        return gpu


def park(modules: list[nn.Module], swapper: BlockSwapper | list[BlockSwapper] | None = None) -> list:
    """Move every GPU parameter and buffer of ``modules`` to system RAM (the frozen tensors of blocks the
    swappers stream are already there and are left alone). Trainable parameters (a LoRA being trained)
    move in place, so an optimizer still has them. Returns what ``unpark`` needs to put them back."""
    skip = set()
    for sw in swapper if isinstance(swapper, list) else [swapper] if swapper is not None else []:
        sw.drop_prefetched()
        sw.release_all()
        for i in sw.swapped:
            skip.update(id(m) for m in sw.blocks[i].modules())
    moved, seen = [], set()
    for root in modules:
        for module in root.modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            for kind, store in (("param", module._parameters), ("buffer", module._buffers)):
                for name, t in list(store.items()):
                    if t is None or not t.is_cuda:
                        continue
                    if kind == "param" and t.requires_grad:
                        moved.append((t, "data", None, t.device))
                        t.data = t.data.to("cpu")
                    elif id(module) not in skip:
                        _assign(module, kind, name, t.detach().to("cpu"))
                        moved.append((module, kind, name, t.device))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return moved


def park_optimizer(opt: torch.optim.Optimizer) -> list:
    """Move an optimizer's GPU state to system RAM (while samples render). Returns what ``unpark`` needs."""
    moved = []
    for state in opt.state.values():
        for k, v in state.items():
            if torch.is_tensor(v) and v.is_cuda:
                moved.append((state, "state", k, v.device))
                state[k] = v.to("cpu")
    if moved and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return moved


def unpark(moved: list) -> None:
    for owner, kind, name, device in moved:
        if kind == "data":  # a trainable parameter
            owner.data = owner.data.to(device)
        elif kind == "state":  # an optimizer's
            owner[name] = owner[name].to(device)
        else:
            t = (owner._parameters if kind == "param" else owner._buffers)[name]
            _assign(owner, kind, name, t.to(device))


ROOM_MARGIN = 1e9  # bytes ``make_room`` leaves for estimate error


def make_room(need: int, modules: list[nn.Module], swapper=None, device=None) -> list:
    """Park ``modules`` (see ``park``) if fewer than ``need`` bytes of VRAM are free, else nothing. Returns
    what ``unpark`` needs."""
    if not torch.cuda.is_available():
        return []
    torch.cuda.empty_cache()
    if torch.cuda.mem_get_info(device)[0] >= need + ROOM_MARGIN:
        return []
    return park(modules, swapper)
