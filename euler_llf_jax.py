"""
1D Compressible Euler Equations - Local Lax-Friedrichs (LLF) FVM Solver in JAX
===============================================================================
Equations: ∂U/∂t + ∂F/∂x = 0

Conservative variables: U = [ρ, ρu, E]
  - ρ   : density
  - ρu  : momentum
  - E   : total energy

Fluxes: F = [ρu, ρu² + p, u(E + p)]
  - p = (γ-1)(E - ½ρu²)  : pressure

Scheme: Local Lax-Friedrichs (Rusanov) flux
  F_LLF = ½(F_L + F_R) - ½ α (U_R - U_L)
  where α = max(|u| + c) over L and R states

Time integration: SSP-RK3 (Shu-Osher)
Spatial reconstruction: MUSCL with minmod limiter (2nd order)

Runs on CPU/GPU/TPU via JAX's backend-agnostic XLA compilation.

Test case: Sod shock tube
  Left:  ρ=1.0,  u=0.0, p=1.0
  Right: ρ=0.125, u=0.0, p=0.1
"""

import time
import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Configuration ─────────────────────────────────────────────────────────────
GAMMA = 1.4  # ratio of specific heats
N_CELLS = 10000  # number of grid cells
X_LEFT = 0.0
X_RIGHT = 1.0
T_FINAL = 0.2  # final time
CFL = 0.45  # CFL number
N_GHOST = 2  # ghost cells for MUSCL (2nd order)

# ── Primitive ↔ Conservative conversions ──────────────────────────────────────


def prim_to_cons(rho, u, p):
    """Convert primitive [ρ, u, p] → conservative [ρ, ρu, E]."""
    E = p / (GAMMA - 1.0) + 0.5 * rho * u**2
    return jnp.stack([rho, rho * u, E], axis=-1)


def cons_to_prim(U):
    """Convert conservative U=[ρ, ρu, E] → (ρ, u, p, c)."""
    rho = U[..., 0]
    u = U[..., 1] / rho
    E = U[..., 2]
    p = (GAMMA - 1.0) * (E - 0.5 * rho * u**2)
    c = jnp.sqrt(GAMMA * p / rho)  # sound speed
    return rho, u, p, c


def euler_flux(U):
    """Physical flux F(U) for 1D Euler."""
    rho, u, p, _ = cons_to_prim(U)
    E = U[..., 2]
    return jnp.stack([rho * u, rho * u**2 + p, u * (E + p)], axis=-1)


# ── Slope limiter ──────────────────────────────────────────────────────────────


@jax.jit
def minmod(a, b):
    """Minmod slope limiter (scalar or array)."""
    return jnp.where(a * b > 0.0, jnp.where(jnp.abs(a) < jnp.abs(b), a, b), 0.0)


# ── MUSCL reconstruction ───────────────────────────────────────────────────────


@jax.jit
def muscl_reconstruct(U):
    """
    MUSCL 2nd-order piecewise-linear reconstruction.
    Returns U_L[i], U_R[i] = left/right states at the RIGHT face of cell i.
    Input U has shape (N_cells_with_ghost, 3).
    Output shapes: (N_interfaces, 3) each.
    """
    # Slopes at each cell using minmod of left/right differences
    dU_L = U[1:-1] - U[:-2]  # backward difference
    dU_R = U[2:] - U[1:-1]  # forward difference
    slope = minmod(dU_L, dU_R)

    # Face states: right face of cell i
    U_L = U[1:-1] + 0.5 * slope  # right-biased state of cell i
    U_R = U[1:-1] - 0.5 * slope  # left-biased state of cell i

    # Shift: U_R[i] is left state at interface i+½, U_L[i+1] is right state
    # Interfaces between cells 1..N-2 (interior faces)
    UL_face = U_L[:-1]  # left state at face i+½  (from cell i)
    UR_face = U_R[1:]  # right state at face i+½  (from cell i+1)
    return UL_face, UR_face


# ── LLF (Rusanov) numerical flux ───────────────────────────────────────────────


@jax.jit
def llf_flux(UL, UR):
    """
    Local Lax-Friedrichs / Rusanov flux at each interface.
    UL, UR: shape (N_faces, 3)
    Returns: numerical flux shape (N_faces, 3)
    """
    FL = euler_flux(UL)
    FR = euler_flux(UR)

    _, uL, pL, cL = cons_to_prim(UL)
    _, uR, pR, cR = cons_to_prim(UR)

    # Max wave speed at each interface
    alpha = jnp.maximum(jnp.abs(uL) + cL, jnp.abs(uR) + cR)

    return 0.5 * (FL + FR) - 0.5 * alpha[..., None] * (UR - UL)


# ── Transmissive (zero-gradient) boundary conditions ──────────────────────────


@jax.jit
def apply_bc(U):
    """Extend U with ghost cells (zero-gradient / transmissive BCs)."""
    # U shape: (N,3); pad 2 ghost cells on each side
    left = jnp.stack([U[0], U[0]], axis=0)
    right = jnp.stack([U[-1], U[-1]], axis=0)
    return jnp.concatenate([left, U, right], axis=0)


# ── Spatial residual ───────────────────────────────────────────────────────────


@jax.jit
def spatial_residual(U, dx):
    """
    Compute dU/dt = -1/dx * (F_{i+½} - F_{i-½}).
    U: interior cells (N, 3).
    Returns: RHS shape (N, 3).
    """
    Ug = apply_bc(U)  # (N+4, 3)
    UL, UR = muscl_reconstruct(Ug)  # (N+2, 3) each  [faces -½ .. N+½]
    F = llf_flux(UL, UR)  # (N+2, 3)

    # RHS for interior cells: F[i+1] is face i+½, F[i] is face i-½
    return -(F[1:] - F[:-1]) / dx  # (N, 3)


# ── Time step (CFL condition) ──────────────────────────────────────────────────


@jax.jit
def compute_dt(U, dx):
    """CFL-based time step."""
    _, u, _, c = cons_to_prim(U)
    max_speed = jnp.max(jnp.abs(u) + c)
    return CFL * dx / max_speed


# ── SSP-RK3 time integrator ────────────────────────────────────────────────────


@jax.jit
def ssp_rk3_step(U, dx, dt):
    """
    Shu-Osher SSP-RK3:
      U1 = U  + dt * L(U)
      U2 = ¾U + ¼U1 + ¼dt * L(U1)
      U3 = ⅓U + ⅔U2 + ⅔dt * L(U2)
    """

    def L(V):
        return spatial_residual(V, dx)

    U1 = U + dt * L(U)
    U2 = 0.75 * U + 0.25 * U1 + 0.25 * dt * L(U1)
    U3 = (1.0 / 3.0) * U + (2.0 / 3.0) * U2 + (2.0 / 3.0) * dt * L(U2)
    return U3


# ── Main simulation loop ───────────────────────────────────────────────────────


@jax.jit
def run_simulation_jit(U_init, dx):
    """
    Compiles the ENTIRE time-marching loop into a single GPU executable.
    """
    def cond_fun(state):
        t, step, U = state
        return t < T_FINAL

    def body_fun(state):
        t, step, U = state
        
        # dt remains on the device as a 0-D array
        dt = compute_dt(U, dx)
        
        # Use jnp.minimum instead of Python's min()
        dt = jnp.minimum(dt, T_FINAL - t) 
        
        U_next = ssp_rk3_step(U, dx, dt)
        
        return (t + dt, step + 1, U_next)

    init_state = (jnp.array(0.0), 0, U_init)
    
    # jax.lax.while_loop executes entirely on the GPU
    final_state = lax.while_loop(cond_fun, body_fun, init_state)
    return final_state

def run_sod_shock_tube():
    # ... (Keep your initial condition setup the same) ...
    dx = (X_RIGHT - X_LEFT) / N_CELLS
    x = jnp.linspace(X_LEFT + 0.5 * dx, X_RIGHT - 0.5 * dx, N_CELLS)
    
    rho0 = jnp.where(x < 0.5, 1.0, 0.125)
    u0 = jnp.zeros_like(x)
    p0 = jnp.where(x < 0.5, 1.0, 0.1)
    U = prim_to_cons(rho0, u0, p0)

    print("Compiling and running on GPU...", end=" ", flush=True)
    t0 = time.perf_counter()
    
    # One call. No Python loop overhead. No host-device syncing.
    t_final, step_final, U_final = run_simulation_jit(U, dx)
    
    # Force Python to wait for the GPU to finish everything
    U_final.block_until_ready() 
    
    wall = time.perf_counter() - t0
    print(f"\nDone: {step_final} steps, wall time = {wall:.4f}s")

    return x, U_final

#def run_sod_shock_tube():
#    print(f"JAX backend: {jax.default_backend()}")
#    print(f"Devices: {jax.devices()}")
#    print(f"Grid: {N_CELLS} cells, domain [{X_LEFT}, {X_RIGHT}]")
#    print(f"Final time: {T_FINAL}, CFL: {CFL}\n")
#
#    dx = (X_RIGHT - X_LEFT) / N_CELLS
#    x = jnp.linspace(X_LEFT + 0.5 * dx, X_RIGHT - 0.5 * dx, N_CELLS)
#
#    # Sod initial conditions (discontinuity at x = 0.5)
#    rho0 = jnp.where(x < 0.5, 1.0, 0.125)
#    u0 = jnp.zeros_like(x)
#    p0 = jnp.where(x < 0.5, 1.0, 0.1)
#
#    U = prim_to_cons(rho0, u0, p0)  # (N, 3)
#
#    # JIT-compile the step function (triggers compilation on first call)
#    print("Compiling JIT kernels...", end=" ", flush=True)
#    t0_compile = time.perf_counter()
#
#    dt_init = compute_dt(U, dx)
#    _ = ssp_rk3_step(U, dx, dt_init).block_until_ready()
#
#    t_compile = time.perf_counter() - t0_compile
#    print(f"done in {t_compile:.2f}s\n")
#
#    # Time-march
#    t = 0.0
#    step = 0
#    t0 = time.perf_counter()
#
#    while t < T_FINAL:
#        dt = float(compute_dt(U, dx))
#        dt = min(dt, T_FINAL - t)
#
#        U = ssp_rk3_step(U, dx, dt)
#        t += dt
#        step += 1
#
#        if step % 200 == 0:
#            elapsed = time.perf_counter() - t0
#            print(f"  step={step:5d}  t={t:.5f}  dt={dt:.2e}  wall={elapsed:.2f}s")
#
#    U.block_until_ready()
#    wall = time.perf_counter() - t0
#    print(f"\nDone: {step} steps, wall time = {wall:.3f}s  ({step / wall:.0f} steps/s)")
#
#    return x, U


# ── Exact Sod solution (for comparison) ───────────────────────────────────────


def exact_sod(x_arr, t):
    """Exact Riemann solution for Sod shock tube (γ=1.4)."""
    g = GAMMA
    gm1 = g - 1.0
    gp1 = g + 1.0

    rhoL, uL, pL = 1.0, 0.0, 1.0
    rhoR, uR, pR = 0.125, 0.0, 0.1
    x0 = 0.5

    cL = np.sqrt(g * pL / rhoL)
    cR = np.sqrt(g * pR / rhoR)

    # Solve for post-contact pressure p* iteratively
    from scipy.optimize import brentq

    def f(p_star):
        if p_star > pR:
            # shock on right
            A = 2.0 / (gp1 * rhoR)
            B = gm1 / gp1 * pR
            fR = (p_star - pR) * np.sqrt(A / (p_star + B))
        else:
            fR = 2.0 * cR / gm1 * ((p_star / pR) ** (gm1 / (2.0 * g)) - 1.0)
        # rarefaction on left
        fL = 2.0 * cL / gm1 * ((p_star / pL) ** (gm1 / (2.0 * g)) - 1.0)
        return fL + fR + (uR - uL)

    p_star = brentq(f, 1e-8, max(pL, pR) * 10)

    # Contact velocity
    u_star = uL - 2.0 * cL / gm1 * ((p_star / pL) ** (gm1 / (2.0 * g)) - 1.0)

    # Left rarefaction (fan)
    rho_starL = rhoL * (p_star / pL) ** (1.0 / g)
    c_starL = np.sqrt(g * p_star / rho_starL)

    # Right shock
    rho_starR = rhoR * (p_star / pR + gm1 / gp1) / (gm1 / gp1 * p_star / pR + 1.0)
    shock_spd = uR + cR * np.sqrt((gp1 / (2.0 * g)) * p_star / pR + gm1 / (2.0 * g))

    rho_ex = np.zeros_like(x_arr)
    u_ex = np.zeros_like(x_arr)
    p_ex = np.zeros_like(x_arr)

    for i, xi in enumerate(x_arr):
        s = (xi - x0) / t if t > 0 else 0.0

        if s < uL - cL:  # undisturbed left
            rho_ex[i] = rhoL
            u_ex[i] = uL
            p_ex[i] = pL
        elif s < u_star - c_starL:  # inside rarefaction
            u_e = 2.0 / gp1 * (cL + (xi - x0) / t + gm1 / 2.0 * uL)
            c_e = cL - gm1 / 2.0 * (u_e - uL)
            rho_e = rhoL * (c_e / cL) ** (2.0 / gm1)
            p_e = pL * (rho_e / rhoL) ** g
            rho_ex[i] = rho_e
            u_ex[i] = u_e
            p_ex[i] = p_e
        elif s < u_star:  # left of contact
            rho_ex[i] = rho_starL
            u_ex[i] = u_star
            p_ex[i] = p_star
        elif s < shock_spd:  # right of contact, left of shock
            rho_ex[i] = rho_starR
            u_ex[i] = u_star
            p_ex[i] = p_star
        else:  # undisturbed right
            rho_ex[i] = rhoR
            u_ex[i] = uR
            p_ex[i] = pR

    return rho_ex, u_ex, p_ex


# ── Plotting ───────────────────────────────────────────────────────────────────


def plot_results(x, U):
    x_np = np.array(x)
    rho, u, p, _ = cons_to_prim(U)
    rho_np = np.array(rho)
    u_np = np.array(u)
    p_np = np.array(p)
    e_np = np.array(p / ((GAMMA - 1.0) * rho))  # specific internal energy

    # Exact solution
    rho_ex, u_ex, p_ex = exact_sod(x_np, T_FINAL)
    e_ex = p_ex / ((GAMMA - 1.0) * rho_ex)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Sod Shock Tube — 1D Compressible Euler (LLF-FVM, MUSCL, SSP-RK3)\n"
        f"N={N_CELLS} cells,  t={T_FINAL},  JAX backend: {jax.default_backend()}",
        fontsize=13,
        fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    fields = [
        (rho_np, rho_ex, "Density  ρ", "royalblue"),
        (u_np, u_ex, "Velocity  u", "darkorange"),
        (p_np, p_ex, "Pressure  p", "forestgreen"),
        (e_np, e_ex, "Specific Internal Energy  e", "crimson"),
    ]

    for idx, (num, ex, title, col) in enumerate(fields):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.plot(x_np, ex, "k-", lw=1.5, label="Exact", zorder=3)
        ax.plot(x_np, num, "o", ms=1.5, color=col, label="LLF-FVM", zorder=2)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(X_LEFT, X_RIGHT)

    plt.savefig("./sod_shock_tube.png", dpi=150, bbox_inches="tight")
    print("Plot saved: sod_shock_tube.png")
    plt.show()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Optionally force a backend: jax.config.update("jax_platform_name", "cpu")
    # Enable 64-bit floats for accuracy (optional):
    # jax.config.update("jax_enable_x64", True)

    x, U = run_sod_shock_tube()
    plot_results(x, U)
