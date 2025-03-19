from typing import Optional
from argparse import Namespace

def parse_list_args(
    args      : Namespace, 
    arg_names : list[str],
    arg_types : Optional[list] = None,
):
    """
        Parse str arguments that should be lists.
    """
    if not arg_types:
        arg_types = [float for _ in arg_names]
        
    for arg_name, arg_type in zip(arg_names, arg_types):
        assert hasattr(args, arg_name), f"Argument {arg_name} not found in args"
        arg_value = getattr(args, arg_name)
        if isinstance(arg_value, str):
            arg_value = eval(arg_value)
            assert isinstance(arg_value, list), f"Argument {arg_name} was not evaluated to a list"
            arg_value = [arg_type(_) for _ in arg_value]
        else:
            raise ValueError(f"Argument {arg_name} is not a string")
        setattr(args, arg_name, arg_value)
    
    return args

def attach_args(
    obj    : object,
    kwargs : dict,
):
    """
        Intended use is to attach arguments to an object in __init__.
        
        Example:
            def __init__(self, *args, **kwargs):
                gffx.utils.attach_args(self, locals())
    """
    for k, v in kwargs.items():
        if k == "self":
            continue
        if hasattr(obj, k):
            raise ValueError(f"Object already has attribute {k}")
        setattr(obj, k, v)