# Instructor Key — Hand steps (η=0.1)

## GD one step from w⁽⁰⁾=[0,0], φ=[1,x]
∇MSE(w⁽⁰⁾) = (2/3) * Σ (w·φ(x)-y)φ(x) = [-4.67, -12.67] → w⁽¹⁾ = [0.47, 1.27].

## SGD two updates (order: (1,1) → (2,3)), φ=[1,x]
After (1,1): g=[-2,-2] → w=[0.2,0.2]; After (2,3): g=[-4.8,-9.6] → w=[0.68,1.16].

## Quadratic mini-step on (1,1), φ₂=[1,x,x²], start w=[0,0,0]
g=[-2,-2,-2] → w=[0.2,0.2,0.2].
