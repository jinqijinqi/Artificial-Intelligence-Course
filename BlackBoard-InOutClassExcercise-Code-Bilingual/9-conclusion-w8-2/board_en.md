# Board (EN) — CS221 Conclusion & Roadmap (Paradigms · Tools · Ethics)

## Four modeling paradigms (what to use when)
- **Reflex-based (low-level signals → actions)**  
  *Models*: linear, neural nets, kNN.  *Inference*: feedforward.  *Learning*: SGD / alternating min.  
  Use for perception/instant decisions.  
- **State-based** (summarize past with state \(s\))  
  *Models*: search problems, **MDPs**, adversarial games.  *Inference*: UCS/A\*, DP/value iteration, minimax.  
  *Learning*: structured Perceptron, Q-learning/TD.  
- **Variable-based** (factor graphs capture independence)  
  *Models*: **CSPs**, Markov nets, Bayes nets.  *Inference*: backtracking/AC-3, forward–backward/beam, Gibbs/MCMC.  
  *Learning*: MLE/EM.  
- **Logic-based** (formulas, infinite models)  
  *Models*: propositional & first-order logic.  *Inference*: model checking, **modus ponens**, resolution.  *Learning*: (open).

## ML as loss minimization
\(\min_w \sum_{(x,y)\in D_{\text{train}}} \mathrm{Loss}(x,y;w)\) with SGD: \(w\leftarrow w-\eta_t \nabla \mathrm{Loss}\).

## Tools mindset
Start from the **problem**, pick a simple tool that fits; avoid overengineering.

## Next courses (categories)
- **Methods**: CS229/230, CS228/236, CS329D, CS330 …  
- **Applications**: CS231N/231A/348I (vision&graphics), CS224N/U/V/C/324 (NLP & LLMs), CS237AB/223A (robotics) …  
- **Foundations**: EE364/CS334 (convex opt), STATS200/214/CS229M (stats & theory), PSYCH204/CS428, PSYCH242/APPPHYS293 …

## Ethics: data → models → predictions
- **Data**: web-scrapes carry offensive content & historical bias; consent & labor matter.  
- **Objectives**: misaligned surrogates (clicks) can harm.  
- **Inequality**: audit; consider worst-group loss.  
- **Harmful uses**: dual-use (deepfakes, autonomous weapons).  
- **IA vs AI**: prioritize **intelligence augmentation** over pure automation.  
**Takeaway**: *Can build* ≠ *should build*. Think benefits/risks and context.
