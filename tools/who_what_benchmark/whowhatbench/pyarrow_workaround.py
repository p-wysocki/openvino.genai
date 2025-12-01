import atexit
import gc
import sys

def _shutdown_pyarrow():
    try:
        gc.collect()
        
        try:
            import pyarrow as pa
            if hasattr(pa, 'jemalloc_memory_pool'):
                try:
                    pa.jemalloc_memory_pool()
                except:
                    pass
        except ImportError:
            pass
        
        gc.collect()
        
    except Exception as e:
        pass

# Register the cleanup function to run before Python finalizes
atexit.register(_shutdown_pyarrow)

def _cleanup_on_del():
    try:
        _shutdown_pyarrow()
    except:
        pass

import weakref
_cleanup_ref = weakref.finalize(sys.modules[__name__], _cleanup_on_del)
