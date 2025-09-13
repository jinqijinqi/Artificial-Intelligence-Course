# Class — SAME Problem: Tic-Tac-Toe (Minimax/Alpha–Beta/Eval)

**Representation** Board is 3×3 (row-major string of 9 chars), MAX='X', MIN='O', '.' empty.
**Given position (MAX to move)**: `X.O..O...`

## Tasks
1) **Depth-2 minimax** by hand: enumerate MAX moves, then optimal MIN replies; leaf utilities \(U\in\{+1,0,-1\}\).  
2) **Alpha–Beta trace** with move ordering (center > corner > edge): write α/β updates and pruned branches.  
3) **Depth-limited evaluation (D=3)**: propose \( \mathrm{Eval}(s)=w^\top\phi(s)\) with features:
   - open X two-in-a-row, open O two-in-a-row, center control, #X corners.  
   Give compact formula and evaluate the given position.  
4) **Quiescence**: find a position where static depth-1 eval fails (horizon effect). Propose a tactical extension rule for TTT.
