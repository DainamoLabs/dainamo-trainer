import os

# Nothing here uses torch.compile, but importing diffusers loads bitsandbytes, which starts PyTorch's pool of
# compile workers: ~17 idle processes in the server, each Sandbox worker and each training process. One
# compile thread means no pool. (Set here so it applies before torch reads it; child processes inherit it.)
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
