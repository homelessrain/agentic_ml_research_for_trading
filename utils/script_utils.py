import argparse
import importlib
import re


def _camel_to_snake(name: str) -> str:
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()


def resolve_model_class(model_class_name: str) -> type:
    """Resolve a model class by name from models/ or models/agent/."""
    module_suffix = _camel_to_snake(model_class_name)
    for package in ('models', 'models.agent'):
        module_path = f'{package}.{module_suffix}'
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError:
            continue
        model_class = getattr(module, model_class_name, None)
        if model_class is not None:
            return model_class
    raise ValueError(
        f'Model class {model_class_name!r} not found in models/ or models/agent/'
    )


def auto_cast(value: str) -> int | float | str:
    """Cast a string to int, float, or leave as str — in that order."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def build_param_overrides(raw_params: list[str]) -> dict:
    """Convert ['key=value', ...] into a typed dict."""
    overrides = {}
    for item in raw_params:
        if '=' not in item:
            raise argparse.ArgumentTypeError(
                f"--param must be in KEY=VALUE format, got: {item!r}"
            )
        key, _, val = item.partition('=')
        overrides[key.strip()] = auto_cast(val.strip())
    return overrides