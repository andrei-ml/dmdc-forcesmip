import numpy as np
from eofs.standard import Eof
from sklearn.base import BaseEstimator
from nipals import nipals

class DMDc(BaseEstimator):
    def __init__(self, truncation=20, lag=3):
        self.truncation = truncation
        self.lag = lag

    def __getstate__(self):
        # Return only what you want to save
        state = self.__dict__.copy()
        # Remove unwanted attributes
        for k in ['_pcs','_data_start','_eigs','_evecs','_ld_eig','_ld_evec','_forcings']:
            state.pop(k, None)
        return state

    def __setstate__(self, state):
        # Restore only the saved attributes
        self.__dict__.update(state)
        # Recreate anything that shouldn't be pickled
 
    def fit(self, data: np.array, forcings: np.array):

        self._forcings = forcings

        self.run_eofs(data)
        
        # regression solution
        inputs = np.concatenate([self._pcs[:-self.lag,:], self._forcings[:,None][:-self.lag,:]],axis=-1)
        targets = self._pcs[self.lag:,:]
        G = np.linalg.lstsq(inputs,targets,rcond=None)[0] 
        
        self.A = G[:self.truncation,:].T
        self.B = G[self.truncation:,:].T

    def predict(self, data: np.array = None, forcings: np.array = None):

        if data is not None and forcings is not None:
            if not hasattr(self, 'data_mean') or not hasattr(self, 'eofs'):
                raise RuntimeError("Model must be fit before calling predict with new data.")
            data_centered = data - self.data_mean  # expected shape (time, ...spatial...)
            t = data_centered.shape[0]
            sp_shape = int(np.prod(data_centered.shape[1:]))
            data_flat = data_centered.reshape(t, sp_shape)
            self._pcs = data_flat @ self.eofs.T

            self._forcings = forcings
        elif data is not None or forcings is not None:
            raise ValueError("Both data and forcings must be provided together.")
        
        self._eigs, self._evecs  = np.linalg.eig(self.A)

        self.least_damped_mode()

        # predict forced response
        
        dmdc_pred_forced_resp_pca = np.linalg.multi_dot(
                                [self._ld_evec, 
                                 np.diag(self._ld_eig), 
                                 np.linalg.pinv(self._ld_evec), 
                                 self._pcs[:-self.lag,:].T])  + self.B @ np.expand_dims(self._forcings[:-self.lag],0)

        dmdc_pred_forced_resp_pca = np.hstack([self._pcs[:self.lag,:].T, dmdc_pred_forced_resp_pca])

        dmdc_pred_forced_resp = np.real(dmdc_pred_forced_resp_pca.T @ self.eofs)

        return dmdc_pred_forced_resp, self.data_mean

    def least_damped_mode(self):
        # the purely real eigenvalue with the largest growth rate
        omegas = np.log(self._eigs)/self.lag
        mask = np.isclose(omegas.imag, 0)
        real_vals = np.real(omegas * mask)
        real_vals[real_vals == 0] = -np.inf
        idx = np.argmax(real_vals)

        self._ld_evec = self._evecs[:, idx:idx+1]
        self._ld_eig = self._eigs[idx:idx+1]

    def run_eofs(self, data:np.asfarray):
        nan_flag = np.sum(np.isnan(data))
        if nan_flag:
            pca = nipals.Nipals(data)
            pca.fit(ncomp=self.truncation, center = False, scale = False, maxiter=10000, tol = 0.0001)
            self.eofs = pca.loadings.to_numpy().T
            self._pcs = pca.scores.to_numpy()
            self.data_mean = pca.x_mean
        else:
            self.data_mean = data.mean(axis = 0)
            data_centered = data - self.data_mean
            eofs_xr = Eof(data_centered, center=False, ddof=1)
            self.eofs = eofs_xr.eofs(neofs=self.truncation)
            self._pcs = eofs_xr.pcs(npcs=self.truncation)
