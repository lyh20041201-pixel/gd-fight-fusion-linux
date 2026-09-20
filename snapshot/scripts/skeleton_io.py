"""Bounded retries for Windows readers temporarily denying atomic status-file replacement."""
import time
from scripts.skeleton_common import write as original_write


def write(path,value):
    for attempt in range(10):
        try:
            return original_write(path,value)
        except PermissionError:
            if attempt==9:
                raise
            time.sleep(min(.5,.025*2**attempt))
