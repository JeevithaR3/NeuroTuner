# policy.py
def adapt_config(current_config, metrics, hw_state):
    """
    current_config: dict
    metrics: last run metrics
    hw_state: get_hardware_state()
    """
    cfg = current_config.copy()
    if hw_state["gpu_percent"] > 85 or hw_state["cpu_percent"] > 85:
        # reduce batch size (if possible)
        bs = cfg.get("batch_size", 32)
        cfg["batch_size"] = max(8, bs // 2)
        cfg["note"] = "reduced batch due to high load"
    # If latency too high, try learning rate smaller (example heuristic)
    if metrics.get("latency_ms", 0) > 150:
        cfg["lr"] = max(1e-5, cfg.get("lr", 1e-3) * 0.5)
        cfg["note"] = cfg.get("note", "") + " ; lowered lr to reduce compute"
    return cfg
