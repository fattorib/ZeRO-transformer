""" 
Calculations used for determing normalizing steps
"""

MAX_CONTEXT = 1024
TOTAL_TOKENS = 12574161 * 1024
WARMUP_STAGES = [128, 256]
WARMUP_STEPS_PER_STAGE = 6000
TOTAL_STEPS = 24550
BATCH_SIZE = 512

tokens_seen_warmup = (
    (BATCH_SIZE * WARMUP_STEPS_PER_STAGE * WARMUP_STAGES[0])
    + (BATCH_SIZE * WARMUP_STEPS_PER_STAGE * WARMUP_STAGES[1])
    + (
        BATCH_SIZE
        * (TOTAL_STEPS - len(WARMUP_STAGES) * WARMUP_STEPS_PER_STAGE)
        * MAX_CONTEXT
    )
)
print(
    f"Total tokens seen in one epoch with sequence length warmup: {tokens_seen_warmup/1e9:.2f} B"
)
print(
    f"Total tokens seen in one epoch without sequence length warmup: {TOTAL_TOKENS/1e9:.2f} B"
)
print(
    f"Required extra training steps @ full context: {(TOTAL_TOKENS - tokens_seen_warmup)/(1024*512)}"
)
print(1e7)