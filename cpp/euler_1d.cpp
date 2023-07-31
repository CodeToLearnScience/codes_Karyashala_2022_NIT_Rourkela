#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

template<typename T>
inline constexpr T Gamma = 1.4;

template<class T>
class Grid
{
public:
  Grid(T x_min, T x_max, std::size_t nx, std::size_t n_ghost = 1)
    : m_x_min(x_min)
    , m_x_max(x_max)
    , m_nx(nx)
    , m_n_ghost(n_ghost)
  {
    m_dx = (m_x_max - m_x_min) / (m_nx - 1);
    m_n_total = m_nx + 2 * m_n_ghost;
    m_x.resize(m_n_total, 0);
  }

  auto min_tick_real() -> std::size_t { return m_n_ghost; }

  auto max_tick_real() -> std::size_t { return m_n_total - m_n_ghost; }

  auto min_tick_ghost() -> std::size_t { return min_tick_real() - m_n_ghost; }

  auto max_tick_ghost() -> std::size_t { return max_tick_real() + m_n_ghost; }

  auto dx() -> T { return m_dx; }

  auto grid() -> std::vector<T>
  {
    for (std::size_t i = 1; i < m_n_total; ++i)
      m_x[i] = m_x_min + (i - 1) * m_dx;

    m_x[0] = m_x[1];
    m_x[m_n_total - 1] = m_x[m_n_total - 2];
    return m_x;
  }

private:
  std::size_t m_nx;
  std::size_t m_n_ghost;
  std::size_t m_n_total;
  T m_x_max;
  T m_x_min;
  T m_dx;
  std::vector<T> m_x;
};

template<class T>
class Euler
{
public:
  explicit Euler(const std::array<T, 3>& cons_vars) { m_cons_vars = cons_vars; }

  auto cons_vars() -> std::array<T, 3> { return m_cons_vars; }

  auto prim_vars() -> std::array<T, 3>
  {
    std::array<T, 3> prim_var{};
    prim_var[0] = m_cons_vars[0];
    prim_var[1] = m_cons_vars[1] / m_cons_vars[0];
    const auto E = m_cons_vars[2] / m_cons_vars[0];

    prim_var[2] =
      (E - 0.5 * prim_var[1] * prim_var[1]) * prim_var[0] * (Gamma<T> - 1);

    return prim_var;
  }

  auto flux() -> std::array<T, 3>
  {
    std::array<T, 3> flux{};
    const auto prim_var = prim_vars();
    flux[0] = m_cons_vars[1];
    flux[1] = prim_var[2] + m_cons_vars[1] * prim_var[1];
    flux[2] = prim_var[2] * prim_var[1] + m_cons_vars[2] * prim_var[1];
    return flux;
  }

  auto max_eig_val() -> T
  {
    const auto prim_var = prim_vars();
    const auto a = sqrt(Gamma<T> * prim_var[2] / prim_var[0]);
    return std::fabs(prim_var[1]) + a;
  }

  auto dt(T CFL, T dx) -> T
  {
    const auto max_eig_value = max_eig_val();
    return CFL * dx / max_eig_value;
  }

private:
  std::array<T, 3> m_cons_vars;
};

template<typename T>
auto
prim_to_cons_vars(std::array<T, 3>& prim_vars) -> std::array<T, 3>
{
  std::array<T, 3> cons_vars{};
  cons_vars[0] = prim_vars[0];
  cons_vars[1] = prim_vars[1] * prim_vars[0];
  const auto E = prim_vars[2] / (prim_vars[0] * (Gamma<T> - 1)) +
                 0.5 * prim_vars[1] * prim_vars[1];
  cons_vars[2] = prim_vars[0] * E;

  return cons_vars;
}

template<typename T>
void
initial_condition(Grid<T>& grid,
                  std::vector<std::array<T, 3>>& cons_vars,
                  const T loc,
                  std::array<T, 3> prim_vars_l,
                  std::array<T, 3> prim_vars_r)
{
  const auto cons_vars_l = prim_to_cons_vars(prim_vars_l);
  const auto cons_vars_r = prim_to_cons_vars(prim_vars_r);

  for (std::size_t i = grid.min_tick_ghost(); i < grid.max_tick_ghost(); i++) {
    if (grid.grid()[i] < loc) {
      for (std::size_t j = 0; j < cons_vars[i].size(); j++) {
        cons_vars[i][j] = cons_vars_l[j];
      }
    } else {
      for (std::size_t j = 0; j < cons_vars[i].size(); j++) {
        cons_vars[i][j] = cons_vars_r[j];
      }
    }
  }
}

template<typename T>
void
boundary_condition(std::vector<std::array<T, 3>>& cons_vars)
{
  for (std::size_t j = 0; j < cons_vars[0].size(); j++) {
    cons_vars[0][j] = cons_vars[1][j];
  }

  for (std::size_t j = 0; j < cons_vars[cons_vars.size() - 1].size(); j++) {
    cons_vars[cons_vars.size() - 1][j] = cons_vars[cons_vars.size() - 2][j];
  }
}

template<class T>
class InterfaceFlux
{
  virtual auto average_flux() -> std::array<T, 3> = 0;
  virtual auto diffusive_flux() -> std::array<T, 3> = 0;
};

template<class T>
class LLF : public InterfaceFlux<T>
{
public:
  LLF(std::array<T, 3> cons_var_l, std::array<T, 3> cons_var_r)
    : m_cons_var_l(cons_var_l)
    , m_cons_var_r(cons_var_r)
  {}

  auto average_flux() -> std::array<T, 3> override
  {
    std::array<T, 3> avg_flux{};
    Euler<T> euler_l(m_cons_var_l);
    Euler<T> euler_r(m_cons_var_r);
    for (std::size_t j = 0; j < m_cons_var_r.size(); j++) {
      avg_flux[j] = 0.5 * (euler_l.flux()[j] + euler_r.flux()[j]);
    }
    return avg_flux;
  }

  auto diffusive_flux() -> std::array<T, 3> override
  {
    Euler<T> euler_l(m_cons_var_l);
    Euler<T> euler_r(m_cons_var_l);
    const auto max_eig_val_l = euler_l.max_eig_val();
    const auto max_eig_val_r = euler_r.max_eig_val();

    std::array<T, 3> diff_flux{};
    for (int j = 0; j < 3; ++j) {
      diff_flux[j] = 0.5 * std::max(max_eig_val_l, max_eig_val_r) *
                     (m_cons_var_r[j] - m_cons_var_l[j]);
    }
    return diff_flux;
  }

private:
  std::array<T, 3> m_cons_var_l;
  std::array<T, 3> m_cons_var_r;
};

template<typename T>
void
llf(const std::vector<std::array<T, 3>>& cons_vars,
    std::vector<std::array<T, 3>>& interface_flux)
{
  for (std::size_t i = 1; i < cons_vars.size() - 1; i++) {
    LLF<T> influx_r(cons_vars[i], cons_vars[i + 1]);
    std::array<T, 3> interface_flux_r{};
    for (std::size_t j = 0; j < cons_vars[i].size(); j++) {
      interface_flux_r[j] =
        influx_r.average_flux()[j] - influx_r.diffusive_flux()[j];
    }
    std::array<T, 3> interface_flux_l{};
    LLF<T> influx_l(cons_vars[i - 1], cons_vars[i]);
    for (std::size_t j = 0; j < cons_vars[i].size(); j++) {
      interface_flux_l[j] =
        influx_l.average_flux()[j] - influx_l.diffusive_flux()[j];
    }
    for (std::size_t j = 0; j < cons_vars[i].size(); j++) {
      interface_flux[i][j] = interface_flux_r[j] - interface_flux_l[j];
    }
  }
}

template<typename T>
auto
min_dt(std::vector<std::array<T, 3>>& cons_vars, T CFL, T dx) -> T
{
  T dt = std::numeric_limits<T>::max();
  for (std::size_t i = 1; i < cons_vars.size() - 1; i++) {
    Euler<T> euler(cons_vars[i]);
    dt = std::min(dt, euler.dt(CFL, dx));
  }
  return dt;
}

template<typename T>
void
write_results(const std::vector<std::array<T, 3>>& cons_vars, Grid<T>& grid)
{
  std::ofstream file;
  file.open("results.csv");
  file << "x,rho,u,p"
       << "\n";
  for (std::size_t i = grid.min_tick_real(); i < grid.max_tick_real(); i++) {
    Euler<T> euler(cons_vars[i]);
    const auto prim_var = euler.prim_vars();
    file << grid.grid()[i] << "," << prim_var[0] << "," << prim_var[1] << ","
         << prim_var[2] << "\n";
  }
  file.close();
}

auto
main() -> int
{
  using T = double;
  Grid<T> grid(0.0, 1.0, 1001);
  std::vector<T> x = grid.grid();
  const auto dx = grid.dx();

  std::array<T, 3> prim_vars_l{ 1.0, 0.75, 1.0 };
  std::array<T, 3> prim_vars_r{ 0.125, 0.0, 0.1 };

  std::vector<std::array<T, 3>> cons_vars(x.size());
  std::vector<std::array<T, 3>> cons_vars_new(x.size());
  std::vector<std::array<T, 3>> flux(x.size());
  std::vector<std::array<T, 3>> inter_flux(x.size());
  initial_condition(grid, cons_vars, 0.3, prim_vars_l, prim_vars_r);
  boundary_condition(cons_vars);

  std::cout << grid.min_tick_ghost() << " " << grid.max_tick_ghost() << " "
            << grid.min_tick_real() << " " << grid.max_tick_real() << "\t";

  T time = 0.0;
  while (time < 0.2) {
    const auto dt = min_dt<T>(cons_vars, 0.5, dx);
    llf<T>(cons_vars, inter_flux);
    for (std::size_t i = 1; i < cons_vars.size() - 1; i++) {
      for (std::size_t j = 0; j < cons_vars[i].size(); j++) {
        cons_vars_new[i][j] = cons_vars[i][j] - (dt / dx) * inter_flux[i][j];
      }
    }

    boundary_condition(cons_vars_new);

    for (std::size_t i = 0; i < cons_vars.size(); i++) {
      for (std::size_t j = 0; j < cons_vars[i].size(); j++) {
        cons_vars[i][j] = cons_vars_new[i][j];
      }
    }

    time += dt;
    std::cout << "time: " << time << "\n";
  }

  write_results<T>(cons_vars, grid);

  return 0;
}
