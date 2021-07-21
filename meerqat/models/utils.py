import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def map_if_not_None(values, function, *args, default_value=None, **kwargs):
    """
    Map all not None values through function (along with additionnal arguments)

    Values that are None will output `default_value`

    Parameters
    ----------
    values: list
        of len batch_size
    function: callable
    default_value: optional
        Defaults to None
    *args, **kwargs: additionnal arguments are passed to function

    Returns
    -------
    Output: list
        of len batch_size (same as values), with `default_value` where values are None
    """
    # 1. filter out values that are None
    output = []
    not_None_values, not_None_values_indices = [], []
    for i, value in enumerate(values):
        # will be overwritten for not_None_values
        output.append(default_value)
        if value is not None:
            not_None_values.append(value)
            not_None_values_indices.append(i)
    if not not_None_values:
        return output
        
    # 2. map values that are not None to function
    not_None_output = function(not_None_values, *args, **kwargs)

    # 3. return the results in a list of list with proper indices
    for j, i in enumerate(not_None_values_indices):
        output[i] = not_None_output[j]
    return output
