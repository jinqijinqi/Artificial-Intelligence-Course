# Class — SAME Problem with Constraint → Relaxation → A*

**Original problem (constrained Transportation)**  
State \(s=(\text{loc}, \Delta)\), \(\Delta=\#\text{walk}-\#\text{tram}\ge 0\).  
Start \((1,0)\), End \((n, \Delta\ge 0)\). Actions:
- walk: \( (loc,\Delta)\to(loc+1,\Delta+1)\), cost 1  
- tram: \( (loc,\Delta)\to(2\cdot loc,\Delta-1)\) if \(\Delta-1\ge 0\), cost 2

**Relaxed problem**: drop \(\Delta\ge 0\). State is **location** only.  
Compute \( \mathrm{FutureCost}_{\text{rel}}(\text{loc})\) (DP or UCS on **reversed** relaxed graph).

**Heuristic**: \(h((\text{loc},\Delta)) := \mathrm{FutureCost}_{\text{rel}}(\text{loc})\).

## Tasks
1) Prove \(h\) is **consistent** by the relaxation theorem.  
2) For \(n=30\), list first ~8 pops of A* (show \(g+h\)) and compare to UCS.  
3) Fill a table of \( \mathrm{FutureCost}_{\text{rel}}(\text{loc})\) for loc=1..16.  
4) Bonus: \(h_0=0\), \(h_{\text{walk}}=n-\text{loc}\), show \(h_{\max}=\max(h,h_{\text{walk}})\) consistent.
