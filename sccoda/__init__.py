try:
    from setuptools_scm import get_version

    __version__ = get_version(root="..", relative_to=__file__)
    del get_version
except (LookupError, ImportError):
    try:
        from importlib_metadata import version  # Python < 3.8
    except:
        from importlib.metadata import version  # Python = 3.8
    __version__ = version(__name__)
    del version
