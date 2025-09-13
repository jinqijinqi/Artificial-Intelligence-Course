# Class — SAME Problem: CampusBot (One scenario, many tools)

**Scenario**: Design an on-campus delivery robot *CampusBot* that must (i) detect crosswalks, (ii) plan routes, 
(iii) track pedestrians, (iv) comply with campus rules (no-go zones, time windows), and (v) meet ethical requirements.

Tasks
1) **Tool choice per subtask (justify briefly)**  
   - Crosswalk detection → (reflex-based model + inference + learning)  
   - Route planning under static map → (state-based, search)  
   - Stochastic travel times → (state-based, MDP)  
   - Pedestrian tracking from noisy positions → (variable-based)  
   - Restricted zones/time windows → (logic-based)
2) **Map to algorithms** (pick one each): {linear/CNN/kNN}, {UCS/A\*}, {value iteration}, {forward–backward/Gibbs/particle filter}, {model checking/MP/resolution}.
3) **Ethics checklist** (data/objective/inequality/harmful use/IA): list 1–2 concrete risks & mitigations each.
4) **Course roadmap**: propose a Methods–Applications–Foundations triad of next courses preparing you to ship CampusBot.
Deliverable: a one-pager table with columns **subtask → paradigm → algorithm → why** (+ ethics & roadmap sections).
