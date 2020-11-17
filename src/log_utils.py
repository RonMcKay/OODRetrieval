from sacred.commands import _format_config


def log_config(_run, _log):
    final_config = _run.config
    config_mods = _run.config_modifications
    _log.info(_format_config(final_config, config_mods))