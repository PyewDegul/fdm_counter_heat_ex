import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Set up matrix
for m_0, k in [(1,1), (1,2), (1,4), (2,0.5), (2,1), (2,2)]:

    y1, y2, xi_L = 1, 1, 1
    del_y1, del_y2, del_z = 10**(-1.5), 10**(-1.5), 10**(-4)  # More fine-grained mesh
    m, n1, n2 = int((xi_L/del_z)+1), int((y1/del_y1)+1), int((y2/del_y2)+1)

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # theta_n1, theta_n2, theta_w initialization with boundary conditions
    theta_n1 =  torch.zeros((m, n1), device=device)
    theta_n2 = torch.zeros((m, n2), device=device)
    theta_w = torch.zeros((m, 1), device=device)
    theta_n1[0, :] = 0  # Inlet n1
    theta_n2[m-1, :] = 1  # Inlet n2
    theta_w[0] = 0  # Wall at zeta = 0
    theta_w[m-1] = 1  # Wall at zeta = 1
    theta_n1[m-1, n1-1] = 1  # Outlet n1
    theta_n2[0, n2-1] = 0  # Outlet n2

    # Tolerance
    errormax = 1e-5
    error = 1
    errors = []
    iteration = 0

    def update(theta_n1, theta_n2, theta_w):

        # Update boundary conditions
        theta_w[1:-1, 0] = (k/(1+k)) * theta_n2[1:-1, n2-2] + (1/(1+k)) * theta_n1[1:-1, n1-2]
        theta_n1[:, n1-1] = theta_w[:, 0]
        theta_n2[:, n2-1] = theta_w[:, 0]
        theta_n1[:, 0] = theta_n1[:, 1]
        theta_n2[:, 0] = theta_n2[:, 1]
        
        # Update inner points
        a1 = del_y1 * del_y1
        b1 = 0.75 * (1 - (torch.arange(1, n1-1, device=device) * del_y1) ** 2)
        c1 = del_z

        a2 = del_y2 * del_y2
        b2 = -0.75 * m_0 * (1 - (torch.arange(1, n2-1, device=device) * del_y2) ** 2)
        c2 = del_z

        # Update theta_n1
        theta_n1[1:m, 1:n1-1] = (c1 / (2 * a1 * b1)) * (theta_n1[:m-1, 2:n1] + theta_n1[:m-1, :n1-2]) + \
                                (1 - (c1 / (a1 * b1))) * theta_n1[:m-1, 1:n1-1]
        # Update theta_n2
        theta_n2[:m-1, 1:n2-1] = -(c2 / (2 * a2 * b2)) * (theta_n2[1:m, 2:n2] + theta_n2[1:m, :n2-2]) + \
                                (1 + (c2 / (a2 * b2))) * theta_n2[1:m, 1:n2-1]

        theta_w[1:-1, 0] = (k/(1+k)) * theta_n2[1:-1, n2-2] + (1/(1+k)) * theta_n1[1:-1, n1-2]
        theta_n1[:, n1-1] = theta_w[:, 0]
        theta_n2[:, n2-1] = theta_w[:, 0]
        theta_n1[:, 0] = theta_n1[:, 1]
        theta_n2[:, 0] = theta_n2[:, 1]

        return theta_n1, theta_n2, theta_w

    tic = datetime.now()

    while error > errormax:
        theta_n1_old = theta_n1.clone()
        theta_n2_old = theta_n2.clone()
        theta_w_old = theta_w.clone()
        
        theta_n1, theta_n2, theta_w = update(theta_n1, theta_n2, theta_w)
        
        error1 = torch.max(torch.abs(theta_n1 - theta_n1_old))  
        error2 = torch.max(torch.abs(theta_n2 - theta_n2_old))
        error = torch.max(error1, error2)
        errors.append(error.item())
        iteration += 1
        if iteration % 2500 == 0:
            toc = datetime.now()
            print(f"Iteration: {iteration}, Error: {error}, consumed_time: {toc - tic}")    


    # Save the results to CSV
    xi = torch.linspace(0, xi_L, m).cpu().numpy()
    y1_values = torch.linspace(0, y1, n1).cpu().numpy()
    y2_values = torch.linspace(0, y2, n2).cpu().numpy()

    T_n1 = theta_n1.cpu().numpy()
    T_n2 = theta_n2.cpu().numpy()

    # np.mean(T_n1[-1]) should be same with effectivness
    print(f'mean(m,k): ({m_0}, {k}) : {np.mean(T_n1[-1])}')

    
