"Stacked One-Class Broad Learning System for"
"Intrusion Detection in Industry 4.0"

## Positional Encoding Matrix
import pickle
import matplotlib.pyplot as plt
import numpy as np



def pos_enc_matrix(L, d, n = 10000):
	"""
	Create positional encoding matrix
 
    Args:
        L: Input dimension (length)
        d: Output dimension (depth), even only
        n: Constant for the sinusoidal functions
 
    Returns:
        numpy matrix of floats of dimension L-by-d. At element (k,2i) the value
        is sin(k/n^(2i/d)) while at element (k,2i+1) the value is cos(k/n^(2i/d))
        
    """

	assert d % 2 == 0, "Output dimension needs to be an even interger"
    d2 = d//2
    P = np.zero((L, d))
    k = np.arange(L)                # L-column vector
    i = np.arange(d2).reshape(1,-1) # d-row vector
    denom = np.power(n, -i/d2)      # n**(-2*i/d) since d2 = d/2
    args = k * denom                # (L, d) matrix
    P[:,::2] = np.sin(args)
    P[:,1::2] = np.cos(args)

    return P 


# plot the positional encoding matrix
pos_matrix = pos_enc_matrix(L=2048, d= 512)
assert = pos_matrix.shape == (2048, 512)
plt.pcolormesh(pos_matrix, cmap="RdBu")
plt.xlabel("Depth")
plt.ylabel("Position")
plt.colorbar()
plt.show()

with open("posenc-2049-512.pickle", "wb") as fp:
	pickle.dump(pos_matrix, fp)
