# Instructor Key — Australia Map Coloring (CSP)

## A consistent coloring (one of many)
SA=R, WA=G, NT=B, Q=G, NSW=B, V=G, T=R.

- All neighbor pairs differ (check edges).  
- T is independent; any color works once mainland is fixed.

## Hand backtracking trace (one example)
Order (MRV+degree) often starts at SA.  
- SA=R → prune {R} from {WA,NT,Q,NSW,V};  
- Next choose NT (degree high): NT=B;  
- WA∈{G} (forced), Q∈{G}, NSW∈{B}, V∈{G};  
- T free.

## Notes for grading
- Any valid 3-coloring is full credit.  
- MRV reasoning, domain pruning steps, and identification of T’s independence earn method points.
