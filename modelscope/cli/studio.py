# Copyright (c) Alibaba, Inc. and its affiliates.
"""ModelScope Studio runtime management CLI.

The ``studio`` command is built into ``modelscope_hub`` so that hub-only
installs can manage Studio spaces too, and this package no longer registers a
competing ``studio`` plugin. This module therefore re-exports the hub
implementation, which keeps ``from modelscope.cli.studio import StudioCMD``
working while leaving exactly one implementation to maintain.

Run ``modelscope studio --help`` for the current command surface; it now also
covers ``list``, ``variable``, ``hardware``, ``base-images`` and
``sdk-versions``.
"""
from modelscope_hub.cli.studio import StudioCMD, StudioCommand


def subparser_func(args):
    """Function which will be called for a specific sub parser."""
    return StudioCMD(args)


__all__ = ['StudioCMD', 'StudioCommand', 'subparser_func']
