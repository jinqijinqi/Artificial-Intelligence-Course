# Instructor Key — Transportation (n=10)

## DP table (FutureCost)
Let T(s) = min time from s to 10.
T(10)=0
T(9)=1
T(8)=2
T(7)=3
T(6)=4
T(5)=min(1+T(6)=5, 2+T(10)=2)=2
T(4)=min(1+T(5)=3, 2+T(8)=4)=3
T(3)=min(1+T(4)=4, 2+T(6)=6)=4
T(2)=min(1+T(3)=5, 2+T(4)=5)=5
T(1)=min(1+T(2)=6, 2+T(2)=7)=6

**Optimal cost = 6**, one optimal path: walk×4 to 5, then tram to 10.

## UCS frontier snapshots (first few)
Init: [(1:0)]
Pop 1 → push (2:1), (2:2) via tram (kept best=1)
Pop 2 (cost 1) → push (3:2), (4:3)
Pop 3 (2) → push (4:3), (6:4)
Pop 4 (3) → push (5:4), (8:5)
Pop 5 (4) → push (6:5), (10:6)  → goal reached with cost 6.

Notes:
- BFS/DFID (unit-cost) are **not** cost-optimal when costs differ (1 vs 2), consistent with lecture assumptions.
