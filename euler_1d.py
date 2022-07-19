import numpy as np
import matplotlib.pyplot as plt

class Grid:
    def __init__(self, xmin: float, xmax: float, nx: int, n_ghost: int = 1):
        self.xmin = xmin
        self.xmax = xmax
        self.nx = nx
        self.tot_nx = nx + 2 * n_ghost
        self.dx = (xmax - xmin) / (nx - 1)
        self.x = np.linspace(xmin, xmax, nx)
        self.x = np.insert(self.x, 0, self.x[0])
        self.x = np.insert(self.x, self.x.shape[0], self.x[-1])

def cons_to_prim_vars(cv: np.ndarray) -> np.ndarray:
    pv = np.zeros(cv.shape)
    pv[:, 0] = cv[:, 0]
    pv[:, 1] = cv[:, 1]/cv[:, 0]
    pv[:, 2] = (cv[:, 2]/cv[:, 0] - 0.5*pv[:, 1]**2) * cv[:, 0]*(1.4-1)

    return pv

def prim_to_cons_vars(pv: np.ndarray) -> np.ndarray:
    cv = np.zeros(pv.shape)
    cv[:, 0] = pv[:, 0]
    cv[:, 1] = pv[:, 0]*pv[:, 1]
    cv[:, 2] = pv[:, 0]*(pv[:, 2]/(pv[:, 0]*(1.4-1)) + 0.5*pv[:, 1]**2)

    return cv

def conv_vars_to_flux(cv: np.ndarray) -> np.ndarray:
    pv = cons_to_prim_vars(cv)
    f = np.zeros(cv.shape)
    f[:, 0] = cv[:, 1]
    f[:, 1] = pv[:, 2] + cv[:, 1]*pv[:, 1]
    f[:, 2] = pv[:, 2]*pv[:, 1] + cv[:, 2]*pv[:, 1]

    return f

def time_step(cv: np.ndarray, dx: float, cfl: float = 0.2) -> np.ndarray:
    pv = cons_to_prim_vars(cv)
    a = np.sqrt(1.4*pv[:, 2]/pv[:, 0])
    eig_val = np.abs(pv[:, 1] + a)
    dt = cfl*dx/eig_val
    return np.min(dt)


def max_eig_val(cv: np.ndarray) -> np.ndarray:
    pv = cons_to_prim_vars(cv)
    a = np.sqrt(1.4*pv[:, 2]/pv[:, 0])
    eig_val = np.abs(pv[:, 1] + a)
    return np.max(eig_val)


def llf_flux(cv: np.ndarray) -> np.ndarray:
    f = conv_vars_to_flux(cv)
    pv = cons_to_prim_vars(cv)
    a = np.sqrt(1.4*pv[:, 2]/pv[:, 0])
    eig_val_l = np.abs(pv[1:-1, 1]) + a[1:-1]
    eig_val_r = np.abs(pv[2:, 1]) + a[2:]
    max_eig_val = np.maximum(eig_val_l, eig_val_r)
    max_eig_val = max_eig_val.reshape(max_eig_val.size, 1)
    f_iph = 0.5*(f[1:-1, :] + f[2:, :]) - 0.5 * \
        max_eig_val*(cv[2:, :] - cv[1:-1, :])
        
    eig_val_l = np.abs(pv[:-2, 1]) + a[:-2]
    eig_val_r = np.abs(pv[1:-1, 1]) + a[1:-1]
    max_eig_val = np.maximum(eig_val_l, eig_val_r)
    max_eig_val = max_eig_val.reshape(max_eig_val.size, 1)
    f_imh = 0.5*(f[1:-1, :] + f[:-2, :]) - 0.5 * \
        max_eig_val*(cv[1:-1, :] - cv[:-2, :])
    
    return f_iph - f_imh

def initialize(pv_l:np.ndarray, pv_r:np.ndarray, x:np.ndarray, pv: np.ndarray, xloc)-> np.ndarray:
    prim_var[np.argwhere(x<=xloc),:]=pv_l
    prim_var[np.argwhere(x>xloc),:]=pv_r
    
def bc(cv:np.ndarray)->np.ndarray:
    pv = cons_to_prim_vars(cv)
    pv = np.insert(pv, 0, pv[0], axis=0)
    pv = np.insert(pv, pv.shape[0], pv[-1],axis=0)
       
    con_var = prim_to_cons_vars(pv)
    
    return con_var
                  
pv_l = np.array([1.0, 0.75, 1.0])
pv_r = np.array([0.125, 0.0, 0.1])

cfl = 0.9
grid = Grid(0,1,1001)
x=grid.x
xloc = 0.3
prim_var = np.zeros((x.size, 3))
initialize(pv_l, pv_r, x, prim_var, xloc)

cv = prim_to_cons_vars(prim_var)
flux = conv_vars_to_flux(cv)
pv = cons_to_prim_vars(cv)

time = 0

while time <= 0.2:
    dt = time_step(cv, grid.dx, cfl)
    flux=llf_flux(cv)
    cv_new = cv[1:-1,:] - (dt/grid.dx)*flux
    cv = bc(cv_new)
    time+=dt
    print(time)    

pv=cons_to_prim_vars(cv)
analytical = np.loadtxt('/mnt/external_data//my_codes/compressible_euler_1d/output/shock_tube.dat')
plt.plot(x, pv[:,0], label='LLF')
plt.plot(analytical[:,0],analytical[:,1], label='Analytical')
plt.xlabel('x', fontsize=16)
plt.ylabel(r'$\rho $', fontsize=16)
plt.legend(fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)