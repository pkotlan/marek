import sys
from pathlib import Path


def get_asset_path(asset_name: str) -> Path:
    """Get absolute path to asset, works from installed and dev environments."""
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)  # ty:ignore[unresolved-attribute]
    else:
        base = Path(__file__).parent.parent.parent
    return base / "assets" / asset_name
