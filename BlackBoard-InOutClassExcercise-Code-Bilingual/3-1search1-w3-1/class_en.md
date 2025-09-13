# Class — SAME Problem: Transportation (1 → n)

**Problem (modeling)**  
Street blocks 1..n. From state \(s\):
- **walk** to \(s+1\) with cost 1
- **tram** to \(2s\) with cost 2 (only if \(2s\le n\))
Start \(s=1\); **end** if \(s=n\). States only increase ⇒ **acyclic**.

## Tasks
1) **Model**: write Start, IsEnd, Actions, Succ, Cost.  
2) **Tree search sketch**: list reachable states up to depth 3 from \(s=1\). What are \(b\), upper bound \(D\), and a plausible solution depth \(d\)?  
3) **DP recurrence**: derive FutureCost(s). Fill a table for \(n=10\) from \(s=10\downarrow 1\).  
4) **UCS by hand (first few steps)** for \(n=10\): show the frontier (state : pastCost) after each pop until first reaching \(n\). Optimal path and cost?

**Submission**: formulas + your table + UCS snapshots (2–3 decimals).
