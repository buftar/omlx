# Deferred Items — Phase 01 Linalg Foundation

## Out-of-Scope Discoveries

### 1. scheduler.py runtime kwargs incompatibility (pre-existing)
- **File:** `omlx/scheduler.py` lines 1508-1518, 1585-1596
- **Issue:** `make_logits_processors()` is called with `presence_penalty=` and `frequency_penalty=` keyword arguments, but the installed mlx_lm version's `make_logits_processors` signature only accepts `(logit_bias, repetition_penalty, repetition_context_size)`. These calls will fail at runtime when processing requests with non-zero presence/frequency penalties.
- **Why deferred:** Pre-existing issue unrelated to compression/linalg work. Requires architectural understanding of how mlx_lm penalty handling changed.
- **Discovered during:** Phase 01, Task 2 (scheduler import fix)
