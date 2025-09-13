# 作业 — 同题程序实现：逻辑 II 工具包

实现轻量工具包：

1) **命题 CNF + 归结**
   - AST → CNF（去 ↔/→、推 ¬、分配）；  
   - 归结反证 `entails_via_resolution(KB, f)`：返回推导轨迹（父子句→结论），若失败则返回反例描述。

2) **Horn 前向链**
   - 规则 `(premises -> head)` 与事实；导出所有可推原子及其**证明 DAG**。

3) **一阶合一 + 一阶 MP**
   - `Const/Var/Fun` 与 `Pred(name,args)`；  
   - `unify(a,b)`（含基本 occurs-check）、`subst(theta,obj)`；  
   - `fo_modus_ponens(facts, rule)`：用最普遍合一推新事实。

4) **在课堂 KB 上验证**
   - 证明：(i) FC 推得 `Knows(alice,mdp)`；(ii) CNF+归结在 KB∪{¬Knows(alice,mdp)} 上导出空子句；(iii) 一阶 MP 在不命题化的前提下推出结论。

（**选做**）实现 FO-CNF（Skolem 化）与一次一阶归结。

**提交**：代码 + ≤2 页（CNF 步骤、归结轨迹、FC 图、FO-MP 合一）。
