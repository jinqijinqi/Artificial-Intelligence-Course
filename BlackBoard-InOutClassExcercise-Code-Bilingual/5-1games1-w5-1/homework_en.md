# Homework — SAME Problem Programmatically: Minimax, Alpha–Beta & Eval on Tic-Tac-Toe

Implement:
1) **Game API**: `legal_moves`, `next_state`, `is_terminal`, `winner`, pretty-print.  
2) **minimax(s, depth)** with node counts; **alphabeta(s, depth)** with ordering (center>corner>edge) + transposition table.  
3) **Evaluation** \(w^\top\phi\): (open-X-2s, open-O-2s, centerX, cornerX). Do grid search for w to maximize win rate vs depth-2 minimax.  
4) **Experiments**: on 50 random mid-game states, compare expanded nodes and αβ speedup across depths; plot depth vs nodes.  
5) **(Optional)** iterative deepening with time budget; killer-move ordering.
