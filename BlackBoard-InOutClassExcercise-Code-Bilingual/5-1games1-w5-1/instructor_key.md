# Instructor Key — Tic-Tac-Toe (given `X.O..O...`, MAX to move)

- **Depth-2 minimax**: Show leaf utilities; optimal play trends to draw; some MAX moves create immediate threats MIN must block.  
- **Alpha–Beta** (center>corner>edge): early good bounds lead to pruning several edge branches. Include a concrete α/β table for one branch.  
- **Eval (D=3)** example: \(\mathrm{Eval}=3\cdot \#\text{open X-2s}-3\cdot \#\text{open O-2s} + 1\cdot \mathbf{1}[s_5=X] + \#\text{X corners}\).  
- **Quiescence**: extend if “current player has/create/stop immediate two-in-a-row”. This fixes simple horizon artifacts in TTT.
