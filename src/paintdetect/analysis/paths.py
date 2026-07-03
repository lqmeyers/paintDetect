"""Small string-based path helpers (from the repo-root ``utils.py``).

Kept importable because notebooks use them. Prefer ``os.path`` in new code.
"""


def getPath(file):
    """Return just the directory portion of a path string."""
    strOut = ''
    i = 1
    while file[-i] != '/':
        i = i + 1
    strOut = file[0:len(file) - (i - 1)]
    return strOut


def getName(file):
    """Return just the filename portion of a path string."""
    strOut = ''
    i = 1
    while file[-i] != '/':
        i = i + 1
    strOut = file[-(i - 1):]
    return strOut
