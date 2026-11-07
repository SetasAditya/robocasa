"""LabApt challenge tasks. Call register() before robosuite.make()."""
from .catalog import VERSION, PROFILES, recipe


def register():
    from . import environments  # noqa: F401
