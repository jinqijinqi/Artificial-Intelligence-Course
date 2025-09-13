# 课堂练习 — 同题：学生–课程–掌握

**域**  
常量：`alice, bob, cs221, mdp`；谓词：`Takes(x,y)`, `Course(y)`, `Covers(y,z)`, `Knows(x,z)`。

**知识库（Horn）**  
1) ∀x∀y∀z  (Takes(x,y) ∧ Covers(y,z)) → Knows(x,z)  
2) Takes(alice, cs221)  
3) Covers(cs221, mdp)  
4) Course(cs221)

### 任务
A) **命题化（Horn）+ MP 完备性**  
- 将变量替换为常量，得到命题原子，如 `Takes_alice_cs221`；  
- 用**前向链（仅 MP）**推导 `Knows_alice_mdp`，画出小型推导 DAG。

B) **CNF 转换（演练）**  
把 \((A∧B)→(C∨D)\) 转为合取范式；并展示 Horn 规则对应的一条子句。

C) **命题归结（非 Horn 附加）**  
在 KB 中加入 `¬Knows(alice, mdp)`，转成 CNF，用归结推出**空子句** \(\Box\)。

D) **一阶 MP + 合一**  
不做命题化，给出前提 `{Takes(alice,cs221), Covers(cs221,mdp)}` 与规则 (1) 的**合一** θ，并推 `Knows(alice,mdp)`。

E)（**选做：一阶归结**）  
由子句 `[¬Takes(x,y) ∨ ¬Covers(y,z) ∨ Knows(x,z)]` 与 `[Takes(alice,cs221)]`、`[Covers(cs221,mdp)]` 做一次**归结**得到实例化结论。
