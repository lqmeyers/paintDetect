"""Experiment config loading.

The config YAML is the single source of truth for a run (``model_settings``,
``train_settings``, ``data_settings``, ``eval_settings``). This loader also
normalizes a known typo in older configs where the eval threshold key was
written as ``out_threshold=`` (stray ``=``).
"""

import yaml


def load_config(config_file):
    """Load and lightly normalize an experiment config YAML into a dict."""
    with open(config_file) as f:
        config = yaml.safe_load(f)

    # Normalize the historical typo `out_threshold=:` -> `out_threshold`.
    eval_cfg = config.get('eval_settings') or {}
    if 'out_threshold=' in eval_cfg and 'out_threshold' not in eval_cfg:
        eval_cfg['out_threshold'] = eval_cfg.pop('out_threshold=')

    return config
