using Dates
using DelimitedFiles
using Plots
using Profile
using ProfileView
using Distributed
using DistributedArrays

mutable struct Grid
  xmin::Float64
  xmax::Float64
  nx::Int64
  n_ghost::Int64
  tot_nx::Int64
  dx::Float64 
  x::Vector{Float64}
  
  function Grid(xmin, xmax, nx)
      new(xmin, xmax, nx, 1, 0, 0.0, zeros(nx))
  end
end

function tot_nx(grid::Grid)
  grid.tot_nx=grid.nx+2*grid.n_ghost    
end

function get_dx(grid::Grid)
  grid.dx=(grid.xmax-grid.xmin)/(grid.nx-1)    
end

function get_grid(grid::Grid)
  resize!(grid.x, grid.tot_nx)
  for i=2:grid.tot_nx-1
      grid.x[i]=(i-2)*grid.dx
  end
  grid.x[1]=grid.x[2]
  grid.x[end]=grid.x[end-1]
end

function prim_to_cons_vars(pv::Vector{Float64}) 
  cv=zeros(Float64, 3)
  cv[1] = pv[1]
  cv[2] = pv[1] * pv[2]
  cv[3] = pv[1] * (pv[3] / (pv[1] * (1.4 - 1.0)) + 0.5 * pv[2] ^ 2)
return cv
end

function cons_to_prim_vars(cv::Vector{Float64})
  pv=zeros(Float64, 3)
  pv[1] = cv[1]
  pv[2] = cv[2] / cv[1]
  pv[3] = (cv[3] / cv[1] - 0.5 * pv[2] ^ 2) * cv[1] * (1.4 - 1.0) 
return pv
end

function conv_vars_to_flux(cv::Vector{Float64})
  pv = cons_to_prim_vars(cv)
  f = zeros(Float64, 3)
  f[1] = cv[2]
  f[2] = pv[3] + cv[2] * pv[2]
  f[3] = pv[3] * pv[2] + cv[3] * pv[2]
  return f
end

function time_step(cv::Matrix{Float64}, dx::Float64, cfl::Float64 = 0.2)
  print(cv)
  dt_min=1.0e32  
  for i=1:size(cv)[1]
      pv = cons_to_prim_vars(cv[i,:])
      a = sqrt(1.4 * pv[3] / pv[1])
      eig_val = abs(pv[2] + a)
      dt = cfl * dx / eig_val
      if dt < dt_min
        dt_min=dt
      end
    end
    return dt_min
end

"""
    llf_flux(cv:: Matrix{Float64})

TBW
"""
function llf_flux(cv:: Matrix{Float64})
    # pv=zeros(Float64, size(cv))
    f_i = zeros(Float64, size(cv))
    f_if = zeros(Float64, size(cv))
    
    for i=1:size(cv)[1]-1
      fl = conv_vars_to_flux(cv[i,:])
      pv = cons_to_prim_vars(cv[i,:])
      a = sqrt(1.4 * pv[3] / pv[1])
      eig_val_l = abs(pv[2]) + a
      fr = conv_vars_to_flux(cv[i+1,:])
      pv = cons_to_prim_vars(cv[i+1,:])
      a = sqrt(1.4 * pv[3] / pv[1])
      eig_val_r = abs(pv[2]) + a
      max_eig_val = max(eig_val_l, eig_val_r)
      f_i[i, :] = 0.5 * (fl + fr) - 0.5 * max_eig_val * (cv[i+1, :] - cv[i, :])
    end

    for i=2:size(cv)[1]-1
        f_if[i, :] = f_i[i, :]-f_i[i-1, :]
    end 

    return f_if
end

function initialize(grid::Vector{Float64}, xloc::Float64, prim_var_l::Vector{Float64}, 
  prim_var_r::Vector{Float64}, cv::Matrix{Float64})
  cv_l=prim_to_cons_vars(prim_var_l)
  cv_r=prim_to_cons_vars(prim_var_r)
  for i=1:size(cv)[1]
      if (grid[i]<xloc)
          cv[i,:] = cv_l
      else
          cv[i,:] = cv_r
      end
  end    
end

function boundary_conditions!(cv::Matrix{Float64})
  # pv=zeros(Float64, size(cv))
  # pv = cons_to_prim_vars!(cv, pv)
  cv[1, :] = cv[2, :]
  cv[end, :] = cv[end-1, :]
  # return prim_to_cons_vars!(pv, cv)
end

"""
    run!(cv)

TBW
"""
function run!(cv::Matrix{Float64}, dx::Float64, cfl::Float64)
    global time = 0.0
    cv_new=zeros(Float64, size(cv))
    while time <= 0.2
        t_dt = time_step(cv, dx, cfl)
        flux = llf_flux(cv)
        for i=2:size(cv)[1]-1
            cv_new[i, :] = cv[i, :] - (t_dt / grid.dx) * flux[i, :]
        end
        cv = boundary_conditions!(cv_new)
        global time += t_dt
    end 
  return cv   
end

grid = Grid(0, 1, 11)
tot_nx(grid)
get_dx(grid)
get_grid(grid)

prim_var_l = Vector{Float64}(undef, 3)
prim_var_r = Vector{Float64}(undef, 3)
prim_var_l = [1.0, 0.75, 1.0]
prim_var_r = [0.125, 0.0, 0.1]
# prim_vars = Matrix{Float64}(undef, grid.tot_nx, 3)
global cons_vars = Matrix{Float64}(undef, grid.tot_nx, 3)
cfl_num=0.9
initialize(grid.x, 0.3, prim_var_l, prim_var_r, cons_vars)

start_time = now()
# addprocs(4)
# @sync @everywhere workers()
# @everywhere using DistributedArrays
# d_cons_vars=distribute(cons_vars)
@time cons_vars=run!(cons_vars, grid.dx, cfl_num) 
# ProfileView.@profview cons_vars=run(cons_vars, grid.dx, cfl_num) 
end_time = now()
println("Total time taken for the simulation is: ", canonicalize(end_time-start_time))
