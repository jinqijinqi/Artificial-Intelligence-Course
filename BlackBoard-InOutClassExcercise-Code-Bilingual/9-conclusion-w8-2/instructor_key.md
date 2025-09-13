# Instructor Key — CampusBot (Paradigms · Algorithms · Ethics · Roadmap)

**Q1 Tooling**  
- Crosswalks → *Reflex*: CNN or linear classifier; inference feedforward; learning by SGD.  
- Static routing → *State*: A\* (if edge costs nonnegative) or UCS.  
- Stochastic costs → *State/MDP*: value iteration / policy iteration; model-based planning.  
- Tracking → *Variable*: HMM (forward–backward) or particle filter; if pairwise factors, Gibbs.  
- Rules → *Logic*: encode as predicates/constraints; check by model checking or MP; conflicts by resolution/SAT.

**Q2 Algorithms**: Example selection — CNN + A\* + value iteration + particle filter + MP.

**Q3 Ethics** (sample risks/mitigations)  
- Data: scraped images may embed bias → curate; datasheets; human review.  
- Objective: avoid pure ETA or clicks → include safety/compliance penalties; multi-objective tuning.  
- Inequality: accessibility (wheelchair users) → worst-group audit; collect data in edge cases.  
- Harmful use: adversarial spoofing/deepfakes of signs → robust training; watermarking; access control.  
- IA: provide operator control & explanations; safe fallback.

**Q4 Course triad (sample)**  
- Methods: CS229, CS230, CS234.  
- Applications: CS237AB, CS223A.  
- Foundations: EE364/CS334, STATS200.

**Code demos**: A\* shortest path on grid; HMM smoothing on evidence [0,2,2]; checklist flags and mitigations.
