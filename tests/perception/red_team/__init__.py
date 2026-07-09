"""Red-team adversarial verification of the LOB-Core C++<->Python zero-copy bridge.

These tests do NOT trust the bridge engineer's summary. They independently try to
break the SPSC lock-free ring (lob_bridge): data corruption, mis-ordering, buffer
overflow past capacity, and torn reads on the atomics under concurrent push/pop.

If the compiled extension (build/lob_bridge*.so) is not importable, every test
SKIPS with a clear message so the regression gate stays meaningful rather than
silently passing on a broken build.
"""
