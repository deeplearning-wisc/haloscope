"""Minimal Baukit tracing helpers without importing Baukit's vision modules."""

import contextlib
import inspect
from collections import OrderedDict

import torch


def get_module(model, name):
    """Resolve a dotted module name from a PyTorch model."""
    for module_name, module in model.named_modules():
        if module_name == name:
            return module
    raise LookupError(name)


def recursive_copy(value, clone=None, detach=None, retain_grad=None):
    """Copy tensor containers according to tracing retention options."""
    if not clone and not detach and not retain_grad:
        return value
    if isinstance(value, torch.Tensor):
        if retain_grad:
            if not value.requires_grad:
                value.requires_grad = True
            value.retain_grad()
        elif detach:
            value = value.detach()
        if clone:
            value = value.clone()
        return value
    if isinstance(value, dict):
        return type(value)(
            {
                key: recursive_copy(
                    item,
                    clone=clone,
                    detach=detach,
                    retain_grad=retain_grad,
                )
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return type(value)(
            recursive_copy(
                item,
                clone=clone,
                detach=detach,
                retain_grad=retain_grad,
            )
            for item in value
        )
    raise TypeError(f"Unknown type {type(value)} cannot be broken into tensors")


def invoke_with_optional_args(function, *args, **kwargs):
    """Invoke a hook with the arguments its signature can accept."""
    argspec = inspect.getfullargspec(function)
    pass_args = []
    used_keywords = set()
    unmatched_positions = []
    used_positions = 0
    defaulted_position = len(argspec.args) - (
        0 if not argspec.defaults else len(argspec.defaults)
    )
    for index, name in enumerate(argspec.args):
        if name in kwargs:
            pass_args.append(kwargs[name])
            used_keywords.add(name)
        elif used_positions < len(args):
            pass_args.append(args[used_positions])
            used_positions += 1
        else:
            unmatched_positions.append(len(pass_args))
            pass_args.append(
                None
                if index < defaulted_position
                else argspec.defaults[index - defaulted_position]
            )
    for name, value in kwargs.items():
        if not unmatched_positions:
            break
        if name in used_keywords or name in argspec.kwonlyargs:
            continue
        pass_args[unmatched_positions.pop(0)] = value
        used_keywords.add(name)
    if unmatched_positions and unmatched_positions[0] < defaulted_position:
        unpassed = ", ".join(
            argspec.args[position]
            for position in unmatched_positions
            if position < defaulted_position
        )
        raise TypeError(f"{function.__name__}() cannot be passed {unpassed}.")
    pass_keywords = {
        name: value
        for name, value in kwargs.items()
        if name not in used_keywords
        and (name in argspec.kwonlyargs or argspec.varkw is not None)
    }
    if argspec.varargs is not None:
        pass_args += list(args[used_positions:])
    return function(*pass_args, **pass_keywords)


class StopForward(Exception):
    """Internal exception used to stop a traced forward pass early."""


class Trace(contextlib.AbstractContextManager):
    """Retain or edit the input/output of one named PyTorch module."""

    def __init__(
        self,
        module,
        layer=None,
        retain_output=True,
        retain_input=False,
        clone=False,
        detach=False,
        retain_grad=False,
        edit_output=None,
        stop=False,
    ):
        self.layer = layer
        if layer is not None:
            module = get_module(module, layer)

        def retain_hook(_module, inputs, output):
            if edit_output:
                output = invoke_with_optional_args(
                    edit_output,
                    output=output,
                    layer=self.layer,
                    inputs=inputs,
                )
            if retain_input:
                self.input = recursive_copy(
                    inputs[0] if len(inputs) == 1 else inputs,
                    clone=clone,
                    detach=detach,
                    retain_grad=False,
                )
            if retain_output:
                self.output = recursive_copy(
                    output,
                    clone=clone,
                    detach=detach,
                    retain_grad=retain_grad,
                )
                if retain_grad:
                    output = recursive_copy(self.output, clone=True, detach=False)
            if stop:
                raise StopForward()
            return output

        self.registered_hook = module.register_forward_hook(retain_hook)
        self.stop = stop

    def __enter__(self):
        return self

    def __exit__(self, exception_type, _value, _traceback):
        self.close()
        return bool(
            self.stop
            and exception_type is not None
            and issubclass(exception_type, StopForward)
        )

    def close(self):
        self.registered_hook.remove()


class TraceDict(OrderedDict, contextlib.AbstractContextManager):
    """Retain or edit the inputs/outputs of several named modules."""

    def __init__(
        self,
        module,
        layers=None,
        retain_output=True,
        retain_input=False,
        clone=False,
        detach=False,
        retain_grad=False,
        edit_output=None,
        stop=False,
    ):
        super().__init__()
        self.stop = stop
        unique_layers = list(dict.fromkeys(layers or []))

        def option_for_layer(option, layer):
            return option.get(layer) if isinstance(option, dict) else option

        for index, layer in enumerate(unique_layers):
            self[layer] = Trace(
                module=module,
                layer=layer,
                retain_output=option_for_layer(retain_output, layer),
                retain_input=option_for_layer(retain_input, layer),
                clone=option_for_layer(clone, layer),
                detach=option_for_layer(detach, layer),
                retain_grad=option_for_layer(retain_grad, layer),
                edit_output=option_for_layer(edit_output, layer),
                stop=stop and index == len(unique_layers) - 1,
            )

    def __enter__(self):
        return self

    def __exit__(self, exception_type, _value, _traceback):
        self.close()
        return bool(
            self.stop
            and exception_type is not None
            and issubclass(exception_type, StopForward)
        )

    def close(self):
        for trace in reversed(self.values()):
            trace.close()
