# Instructor Key — Dice Game (γ=1)

## Policy evaluation (closed form)
For π(stay): \(V^{\pi}(\text{in})=\tfrac{1}{3}(4+0)+\tfrac{2}{3}(4+V^{\pi}(\text{in}))\)  
⇒ \(V^{\pi}(\text{in})=12\). For π(quit): \(V^{\pi}(\text{in})=10\). So \(V^*(\text{in})=12\), \(\pi^*(\text{in})=\text{stay}\).

## Iterative traces (first few)
- Policy eval (π=stay): \(V^{(1)}=4\); \(V^{(2)}=\tfrac{28}{3}\approx 9.33\); \(V^{(3)}\approx 11.11\) → 12.
- Value iteration: initial Q_quit=10, Q_stay=4；数步后 Q_stay>10，策略翻转为 stay，固定点 \(V^*=12\)。

## Discounting
For γ<1, \(V^{\pi=\text{stay}}(\text{in})=\sum_{k\ge0} (2/3)^k \, 4\, \gamma^{k+1} = \frac{4\gamma}{1-\tfrac{2}{3}\gamma}\).  
与 10 比较即可得到策略翻转的阈值。
