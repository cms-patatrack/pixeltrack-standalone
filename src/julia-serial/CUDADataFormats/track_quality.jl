"""
ChatGPT:

Bad: Tracks that are unreliable or cannot be used for analysis due to poor fit or other issues.

Duplicate (dup): Tracks that are duplicates of each other, typically identified as multiple reconstructions of the same physical track due to overlapping hits or detector noise.

Loose: Tracks that pass basic quality criteria but may not meet stricter requirements. These tracks are generally used for preliminary analyses.

Strict: Tracks that meet a stringent set of quality criteria, indicating high confidence in their accuracy. These are used in detailed analyses and precision measurements.

Tight: Tracks that are very stringent in terms of quality criteria, typically used in scenarios where very high precision is required.

High Purity: Tracks that are identified with a high degree of certainty to be true tracks, often verified through additional checks and validation processes.
"""
@enum Quality begin
    bad = 0
    dup = 1
    loose = 2
    strict = 3
    tight = 4
    highPurity = 5
end