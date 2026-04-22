from importlib import metadata as importlib_metadata


try:
    __version__ = importlib_metadata.version("cyfi")
except importlib_metadata.PackageNotFoundError:
    __version__ = "unknown"
