import numpy as np

cstate = [1,0 , 1, 1]
print(sum(cstate))
probabilities = np.ones( 5 ) * 0.1 / (sum(cstate) + 1)
probabilities[[i for i, val in enumerate(cstate) if val == 0]] = 0
print(probabilities)