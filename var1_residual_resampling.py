

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.api import VAR

plt.ion()

# true var
A = np.array([[0.9, 0.6], [0.01, 0.9]])
Sigma = np.eye(2)

def var_sim(T, A, Sigma):
    assert len(set(A.shape)) == 1
    assert len(set(Sigma.shape)) == 1
    assert A.shape[0] == Sigma.shape[0]

    d = A.shape[0]
    L = np.linalg.cholesky(Sigma)

    # add some extra time to burn the process in
    TT = T + 100

    Y = np.empty((TT, d))
    Y[0] = np.dot(L, np.random.randn(d))
    for i in range(1, TT):
        Y[i] = np.dot(A, Y[i-1]) + np.dot(L, np.random.randn(d))

    return Y[100:]


if __name__ == "__main__":

    np.random.seed(13)

    T = 100
    d = A.shape[0]
    nres_samps = 1000

    Y = var_sim(T, A, Sigma)
    Y -= np.mean(Y, axis=0)

    model = VAR(Y)
    res = model.fit(1)
    A_est = res.params[1:].T

    # 1-step ahead predictions
    Yhat = np.empty_like(Y)
    Yhat[0] = Y[0]
    for i in range(1, Yhat.shape[0]):
        Yhat[i] = res.forecast(Y[i-1:i], 1)[0]

    resid = Y - Yhat
    resid_mean = np.mean(resid[1:], axis=0)
    resid[1:] -= resid_mean
    
    A_rr_dist = np.empty((nres_samps, d, d))
    for ni in range(nres_samps):
        inds = np.random.choice(range(0,T-1), size=T-1, replace=True)

        resid_perm = resid[inds]

        Y_rr = Yhat.copy()
        for i in range(1, Y_rr.shape[0]):
            Y_rr[i] = res.forecast(Y_rr[i-1:i], 1)[0] + resid_perm[i-1]

        model_rr = VAR(Y_rr)
        res_rr = model_rr.fit(1)

        A_rr = res_rr.params[1:].T
        A_rr_dist[ni] = A_rr

    # plot the distributions
    fig, axes = plt.subplots(d, d)
    for i in range(d):
        for j in range(d):
            axes[i,j].hist(A_rr_dist[:,i,j], color='steelblue', label="bootstrap")
            axes[i,j].axvline(x=A[i,j], color='darkred', label="true")
            axes[i,j].axvline(x=A_est[i,j], color='orange', label="est.")
            axes[i,j].set_title("A[%d,%d]" % (i, j))

    plt.legend()
    plt.show()

    se_boot = np.sqrt(np.var(A_rr_dist, axis=0, ddof=1))
    print("boostrap std. errors:")
    print(se_boot)

    print("bootstrap confidence intervals:")
    for i in range(d):
        for j in range(d):
            print("A[%d,%d]:" % (i, j), A_est[i,j], "+-", se_boot[i,j])
