import numpy as np
from eofs.standard import Eof
from nipals import nipals
from sklearn.base import BaseEstimator
from matplotlib import pyplot as plt

class DMDc_opt(BaseEstimator):
    def __init__(self, truncation=20, lag=3, optlag=60):
        self.truncation = truncation
        self.lag = lag
        self.optlag = optlag

    def __getstate__(self):
        # Return only what you want to save
        state = self.__dict__.copy()
        # Remove unwanted attributes
        for k in ['_pcs','__Psi','_Phi','_alphas','_limopts']:
            state.pop(k, None)
        return state

    def __setstate__(self, state):
        # Restore only the saved attributes
        self.__dict__.update(state)
        # Recreate anything that shouldn't be pickled

    def fit(self, data: np.array, forcings: np.array):

        self.run_eofs(data)

        self._pcs = np.hstack([self._pcs, np.expand_dims(forcings, axis = 1)])

        self.M = np.linalg.inv(self._pcs[:-self.lag,:].T @ self._pcs[:-self.lag,:]) @ self._pcs[:-self.lag,:].T @ self._pcs[self.lag:,:]

    def predict(self, data: np.array = None, forcings: np.array = None):

        if data is not None and forcings is not None:
            data_centered = data - self.data_mean
            pcs = data_centered @ self.eofs/self.svals
            self._pcs = np.hstack([pcs, np.expand_dims(forcings, axis = 1)])
        elif data is not None or forcings is not None:
            raise ValueError("Both data and forcings must be provided together.")

        Mnormed = self.M / np.linalg.norm(self.M)
        A = np.linalg.matrix_power(Mnormed, int(self.optlag/self.lag))[:,:-1]

        self._Psi, _ ,self._Phit = np.linalg.svd(A, 0)
        self._Phi = self._Phit.T

        self._alphas = self._pcs @ self._Psi
        self._limopts= self.eofs.T @ self._Phi

        dmd_pred_forced_resp = np.matmul(self._alphas[:,0:1],self._limopts.T[0:1,:])

        return dmd_pred_forced_resp, self.data_mean

    def run_eofs(self, data:np.asfarray):
        nan_flag = np.sum(np.isnan(data))
        if nan_flag:
            pca = nipals.Nipals(data)
            pca.fit(ncomp=self.truncation, center = False, scale = False, maxiter=10000, tol = 0.0001)
            eigs = pca.eig.to_numpy()
            self.svals = np.sqrt(eigs)
            self.eofs = pca.loadings.to_numpy().T
            self._pcs = pca.scores.to_numpy()
            self.data_mean = pca.x_mean
        else:
            self.data_mean = data.mean(axis = 0)
            data_centered = data - self.data_mean
            eofs_xr = Eof(data_centered, center=False, ddof=1)
            self.eofs = eofs_xr.eofs(neofs=self.truncation)
            self._pcs = eofs_xr.pcs(npcs=self.truncation)
            self.svals = np.sqrt(eofs_xr.eigenvalues(neigs=self.truncation))

        if np.any(self.svals == 0):
            raise ValueError("Some singular values are zero, cannot scale eofs and pcs.")
        
        self._pcs = self._pcs @ np.diag(1/self.svals)
        self.eofs = np.diag(self.svals) @ self.eofs
