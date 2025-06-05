import torch
import numpy as np

n = 10
t = 10
G = torch.randn(t-1, n)
g = torch.randn(n)
I = 10
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def solve_dualSGD():
    '''
    theory: v* <- 0-vector
    gradF w/ respect to v: G * (transpose(G) * v) + G * g
    new-v_star <- old-v_star - alpha * gradF
    new-v_star <- max[0-vector, v]
    '''
    # learning rate
    lr = 0.01

    # t may be exclusive, which means we would actually use t instead of t-1.         
    v = torch.zeros(t - 1, device=dev)
    z = torch.zeros(t - 1, device=dev)

    # does not depend on v_star
    Gg = np.dot(G, g)
    #print("precomputed Gg:", Gg.shape)
    for i in range(I):
        # todo: move G * transpose(G) to update per task
        print(f"v_{i}: {v}")

        '''
        G * G^T * v
        '''
        # s = G^T * v
        # print(v.shape, G.shape)
        temp = np.dot(G.T, v) # s ∈ n x 1
        # print(f"Temp shape: {temp.shape}")
                # G * s

        full_product = np.dot(G, temp)
        # print(f"Full product shape: {full_product.shape}")
        #
        # print(f"Full Product, Gg shapes: {full_product.shape}, {Gg.shape}")
        
        gradF = full_product + Gg
        # print(f"gradF shape: {gradF.shape}")
        v -= lr * gradF
        v = torch.max(v, z)
    return v
v_star = solve_dualSGD()
print(v_star)
