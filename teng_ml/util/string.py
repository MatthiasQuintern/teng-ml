import inspect
import torch.optim.lr_scheduler as sd
import re

def fill_and_center(s: str, fill_char="=", length=100):
    rs = fill_char * length
    margin = (length - len(s)) // 2
    if margin > 1:
        rs = f"{fill_char*(margin-1)} {s} {fill_char*(margin-1)}"
        if len(rs) == 99: rs = rs + "="
        assert(len(rs) == 100)
        return rs
    else:
        return s

def class_str(x):
    """
    Return the constructor of the class of x with arguemnts
    """
    name = type(x).__name__
    signature = inspect.signature(type(x))
    params = []
    for param_name, param_value in x.__dict__.items():
        if param_name not in signature.parameters:
            continue
        default_value = signature.parameters[param_name].default
        if param_value != default_value:
            params.append(f"{param_name}={param_value!r}")
    if params:
        return f"{name}({', '.join(params)})"
    else:
        return name


def cleanup_str(s):
    """
    convert to string if necessary and
    if scheduler string:
        remove unnecessary parameters
    """
    if not type(s) == str:
        s = str(s)
    # check if scheduler string
    re_scheduler = r"(\w+)\((.*)(optimizer=[A-Za-z]+) \(.*(initial_lr: [\d.]+).*?\)(.*)\)"
    # groups: (sched_name, sched_params1, optimizer=Name, initial_lr: digits, sched_params2)
    match = re.fullmatch(re_scheduler, s.replace("\n", " "))
    if match:
        g = match.groups()
        s = f"{g[0]}({g[1]}{g[2]}({g[3]}, ...){g[4]})"
        return s
    return s
