# Part of the implementation is borrowed and modified from PackNet-SfM,
# made publicly available under the MIT License at https://github.com/TRI-ML/packnet-sfm
from termcolor import colored

from modelscope.models.cv.video_depth_estimation.utils.types import is_list

########################################################################################################################


def filter_dict(dictionary, keywords):
    """
    Returns only the keywords that are part of a dictionary

    Parameters
    ----------
    dictionary : dict
        Dictionary for filtering
    keywords : list of str
        Keywords that will be filtered

    Returns
    -------
    keywords : list of str
        List containing the keywords that are keys in dictionary
    """
    return [key for key in keywords if key in dictionary]


########################################################################################################################


def make_list(var, n=None):
    """
    Wraps the input into a list, and optionally repeats it to be size n

    Parameters
    ----------
    var : Any
        Variable to be wrapped in a list
    n : int
        How much the wrapped variable will be repeated

    Returns
    -------
    var_list : list
        List generated from var
    """
    var = var if is_list(var) else [var]
    if n is None:
        return var
    else:
        assert len(var) == 1 or len(
            var) == n, 'Wrong list length for make_list'
        return var * n if len(var) == 1 else var


########################################################################################################################


def same_shape(shape1, shape2):
    """
    Checks if two shapes are the same

    Parameters
    ----------
    shape1 : tuple
        First shape
    shape2 : tuple
        Second shape

    Returns
    -------
    flag : bool
        True if both shapes are the same (same length and dimensions)
    """
    if len(shape1) != len(shape2):
        return False
    for i in range(len(shape1)):
        if shape1[i] != shape2[i]:
            return False
    return True


########################################################################################################################


def pcolor(string, color, on_color=None, attrs=None):
    """
    Produces a colored string for printing

    Parameters
    ----------
    string : str
        String that will be colored
    color : str
        Color to use
    on_color : str
        Background color to use
    attrs : list of str
        Different attributes for the string

    Returns
    -------
    string: str
        Colored string
    """
    return colored(string, color, on_color, attrs)
