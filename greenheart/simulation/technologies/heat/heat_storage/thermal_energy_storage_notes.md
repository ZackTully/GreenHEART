


### What happens in the low level controller

Inputs: 
- $P_{available}$
- $P_{charge,desired}$
- $Q_{discharge,desired}$

if $P_{charge,desired} > P_{available}$ : $P_{charge, desired} = P_{available}$ and $P_{unused} = 0$
else : $P_{unused} = P_{available} - P_{charge,desired}$

Calculate and saturate
- $m_{charge}$
- $m_{discharge}$

