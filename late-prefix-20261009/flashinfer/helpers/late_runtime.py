"""Read-only resolved worker runtime, reused from the validated free driver."""
import dataclasses
import enum

def jsonable(value):
    if isinstance(value, enum.Enum):
        return value.name
    if dataclasses.is_dataclass(value):
        return {field.name: jsonable(getattr(value, field.name))
                for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    return str(value)


def worker_runtime(worker):
    """Read actual per-rank runtime state through public collective_rpc."""
    import torch
    import torch.distributed as dist
    from vllm.compilation.counter import compilation_counter
    from vllm.distributed.parallel_state import get_tp_group

    runner = worker.model_runner
    config = worker.vllm_config
    manager = runner.cudagraph_manager
    tp = get_tp_group()
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    result = {
        "rank": worker.rank,
        "runner_class": type(runner).__module__ + "." + type(runner).__name__,
        "tp_rank": tp.rank_in_group, "tp_world_size": tp.world_size,
        "tp_device_backend": dist.get_backend(tp.device_group),
        "cuda_device": device, "gpu_name": props.name,
        "gpu_uuid": str(getattr(props, "uuid", "unavailable")),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "nccl_version": list(torch.cuda.nccl.version()),
        "config": {
            name: jsonable(getattr(config, name))
            for name in (
                "model_config", "parallel_config", "cache_config",
                "scheduler_config", "compilation_config", "kernel_config",
                "attention_config", "observability_config", "speculative_config",
            )
        },
        "compilation_counter": jsonable(compilation_counter),
        "graphs_captured": bool(manager and manager._graphs_captured),
        "captured_token_counts": manager.captured_token_counts() if manager else [],
        "graph_descriptors": [jsonable(item) for item in manager.graphs]
        if manager else [],
    }
    return result

