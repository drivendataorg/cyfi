import sys
try:
    from cyfi.version import __version__
    print(f"Version: {__version__}")
except Exception as e:
    print(f"Error getting version: {e}")

try:
    import loguru
    print("Loguru is installed")
except ImportError:
    print("Loguru is NOT installed")
