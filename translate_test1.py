import litert_lm

# Initialize with the model path and optionally specify the backend.
# backend can be Backend.CPU() (default), Backend.GPU() or Backend.NPU().
with litert_lm.Engine(
    "path/to/your/model.litertlm",
    backend=litert_lm.Backend.CPU(),
    # Optional: Pick a writable dir for caching compiled artifacts.
    # cache_dir="/tmp/litert-lm-cache"
) as engine:
    # ... Use the engine to create a conversation ...
    pass