# 作业 — 同题程序实现：井字棋的极大极小、Alpha–Beta 与估值

实现：
1) **博弈接口**：`legal_moves`、`next_state`、`is_terminal`、`winner`、打印函数。  
2) **minimax(s, depth)**（记录节点数）；**alphabeta(s, depth)**（中心>角>边排序 + 置换表）。  
3) **估值** \(w^\top\phi\)：(open-X-2s, open-O-2s, centerX, cornerX)。做网格搜索以提升对 depth-2 minimax 的胜率。  
4) **实验**：在 50 个随机中盘局面对比节点展开数与 αβ 加速比，绘制“深度-节点数”曲线。  
5) **（选做）** 迭代加深（含时限）；killer move 排序。
