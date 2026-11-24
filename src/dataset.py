import re
from pathlib import PurePosixPath

_SUFFIX = re.compile(r"(.+)_t(prev|next)(?:_(\d+)ms)?")


def index_images(names: list[str]) -> dict[str, dict[int, str]]:
    groups = {}
    for name in names:
        path = PurePosixPath(name)
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        base, offset = path.stem, 0
        match = _SUFFIX.fullmatch(path.stem)
        if match:
            base, direction, milliseconds = match.groups()
            offset = int(milliseconds or 1000) * (-1 if direction == "prev" else 1)
        groups.setdefault(path.with_name(base), {})[offset] = name
    return {
        frames[0]: dict(sorted(frames.items()))
        for frames in sorted(groups.values(), key=lambda frames: frames[0])
    }
