import logging


def stats(cfg):
    model = cfg.model

    n_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    model_stats = {
        "n_parameters": n_parameters,
        "n_parameters_m": n_parameters / 1e6,
        "flops": None,
        "macs": None,
    }

    logging.info(
        "Model parameters: %.2fM. FLOPs profiling disabled on SCC.",
        n_parameters / 1e6,
    )

    return n_parameters, model_stats
