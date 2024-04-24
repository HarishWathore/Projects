import os
import warnings

from sympy import Symbol, Eq, Abs

import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.solver import SequentialSolver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_2d import Rectangle, Line, Circle, Channel2D
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.key import Key
from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.utils.io import (
    csv_to_dict,
    ValidatorPlotter,
    InferencerPlotter,
)
from modulus.sym import quantity
import math
import matplotlib.pyplot as plt
import numpy as np
from sympy import Number, Symbol, Heaviside, atan, sin, cos, sqrt,Eq, Abs, tanh, And, Or
import os

from modulus.sym.geometry.primitives_2d import Polygon, Line
from modulus.sym.geometry.parameterization import Parameterization, Parameter
from modulus.sym.utils.io.vtk import var_to_polyvtk
from modulus.sym.geometry import Bounds
import modulus.sym
from modulus.sym.hydra import to_yaml
from modulus.sym.hydra.utils import compose
from modulus.sym.hydra.config import ModulusConfig
from modulus.sym.eq.pdes.basic import NormalDotVec
from modulus.sym.eq.non_dim import NonDimensionalizer, Scaler
import modulus.sym.geometry as geom
from modulus.sym.models.moving_time_window import MovingTimeWindowArch


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:

    # time window parameters
    time_window_size = 1.0
    t_symbol = Symbol("t")
    time_range = {t_symbol: (0, time_window_size)}
    nr_time_windows = 3

    # physical quantities
    nu = quantity(0.0007143, "kg/(m*s)") #RE=1400
    rho = quantity(1.0, "kg/m^3")
    inlet_u = quantity(1.0, "m/s")
    inlet_v = quantity(0.0, "m/s")
    noslip_u = quantity(0.0, "m/s")
    noslip_v = quantity(0.0, "m/s")
    outlet_p = quantity(0.0, "pa")
    velocity_scale = inlet_u
    density_scale = rho
    length_scale = quantity(1, "m")
    nd = NonDimensionalizer(
        length_scale=length_scale,
        time_scale=length_scale / velocity_scale,
        mass_scale=density_scale * (length_scale**3),
    )

    # geometry
    channel_length = (quantity(-10, "m"), quantity(20, "m"))
    channel_width = (quantity(-10, "m"), quantity(10, "m"))
    channel_origin = (0.0,0.0)
    channel_length_nd = tuple(map(lambda x: nd.ndim(x), channel_length))
    channel_width_nd = tuple(map(lambda x: nd.ndim(x), channel_width))

    channel = Channel2D(
        (channel_length_nd[0], channel_width_nd[0]),
        (channel_length_nd[1], channel_width_nd[1]),
    )
    inlet = Line(
        (channel_length_nd[0], channel_width_nd[0]),
        (channel_length_nd[0], channel_width_nd[1]),
        normal=1,
    )
    outlet = Line(
        (channel_length_nd[1], channel_width_nd[0]),
        (channel_length_nd[1], channel_width_nd[1]),
        normal=1,
    )
    wall_top = Line(
        (channel_length_nd[1], channel_width_nd[0]),
        (channel_length_nd[1], channel_width_nd[1]),
        normal=1,
    )
       

    # Given data points for the upper surface

    x_upper = [0.000, 0.000, 0.000, 0.066, 0.124, 0.189, 0.271, 0.341, 0.398, 0.553, 0.610, 0.689, 0.772, 0.833, 0.897, 1.005]
    y_upper = [-0.020,-0.000,0.020, 0.020, 0.058, 0.016, 0.073, 0.013, 0.051, 0.051, 0.015, 0.043, 0.014, 0.010, 0.036, 0.020]

    # Given data points for the lower surface

    x_lower = [0.000, 0.080, 0.128, 0.187, 0.272, 0.337, 0.415, 0.537, 0.606, 0.688, 0.770, 0.839, 0.912, 0.995, 1.005]
    y_lower = [-0.020, -0.020, 0.010, -0.029, 0.023, -0.036, 0.014, 0.014, -0.027, 0.001, -0.028, -0.032, -0.005, -0.020, 0.020]

    # Store upper surface points as a list of tuples
    upper_surface_points = [(x, y) for x, y in zip(x_upper, y_upper)]

    # Store lower surface points as a list of tuples
    lower_surface_points = [(x, y) for x, y in zip(x_lower, y_lower)]

    upper = Polygon(upper_surface_points)

    lower = Polygon(lower_surface_points)

    geo = upper + lower
    
    coro = geo.rotate(angle = 59 * np.pi / 30, axis = "z")

    corrugated_airfoil = channel - coro

    flow_box_origin = (-2.0, -1.0)  # Adjusted to meet the criteria
    flow_box_dim = (5.0, 3.0)       # Adjusted to meet the criteria


    # make list of nodes to unroll graph on
    ns = NavierStokes(nu=nd.ndim(nu), rho=nd.ndim(rho), dim=2, time=True)
    normal_dot_vel = NormalDotVec(["u", "v"])
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("t")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected, 
    )
    time_window_net = MovingTimeWindowArch(flow_net, time_window_size)

    nodes = (
        ns.make_nodes()
        + normal_dot_vel.make_nodes()
        + [time_window_net.make_node(name="time_window_network")]
        + Scaler(
            ["u", "v", "p"],
            ["u_scaled", "v_scaled", "p_scaled"],
            ["m/s", "m/s", "m^2/s^2"],
            nd,
        ).make_node()
    )

    # make initial condition domain
    ic_domain = Domain("initial_conditions")

    # make moving window domain
    window_domain = Domain("window")
    x, y = Symbol("x"), Symbol("y")


    # initial condition
    IC = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=corrugated_airfoil,
        outvar={"u": 0, "v": 0},
        bounds=Bounds({x: channel_length_nd, y: channel_width_nd}),
        batch_size=cfg.batch_size.IC,
        lambda_weighting={"u": 100, "v": 100},
        batch_per_epoch=1000,
        parameterization={t_symbol: 0.0},
    )
    ic_domain.add_constraint(IC, "IC")

    # make constraint for matching previous windows initial condition
    IC = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=corrugated_airfoil,
        outvar={"u_prev_step_diff": 0, "v_prev_step_diff": 0},
        batch_size=cfg.batch_size.ic_interior,
        bounds=Bounds({x: channel_length_nd, y: channel_width_nd}),
        lambda_weighting={
            "u_prev_step_diff": 100,
            "v_prev_step_diff": 100,
            
        },
        batch_per_epoch=1000,

        parameterization={t_symbol: 0.0},
    )
    window_domain.add_constraint(IC, name="IC")


    # inlet
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet,
        outvar={"u": nd.ndim(inlet_u), "v": nd.ndim(inlet_v)},
        batch_size=cfg.batch_size.inlet,
        
        parameterization=time_range,
    )
    ic_domain.add_constraint(inlet, name="inlet")
    window_domain.add_constraint(inlet, name="inlet")


    # outlet
    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet,
        outvar={"p": nd.ndim(outlet_p)},
        batch_size=cfg.batch_size.outlet,
        
        parameterization=time_range,
    )
    ic_domain.add_constraint(outlet, name="outlet")
    window_domain.add_constraint(outlet, name="outlet")


    # full slip (channel walls)
    walls = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=channel,
        outvar={"u": nd.ndim(inlet_u), "v": nd.ndim(inlet_v)},
        batch_size=cfg.batch_size.walls,
        
        parameterization=time_range,
    )

    ic_domain.add_constraint(walls, name="walls")
    window_domain.add_constraint(walls, name="walls")


    # no slip
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=coro,
        outvar={"u": nd.ndim(noslip_u), "v": nd.ndim(noslip_v)},
        batch_size=cfg.batch_size.no_slip,
        
        parameterization=time_range,
    )

    ic_domain.add_constraint(no_slip, name="no_slip")
    window_domain.add_constraint(no_slip, name="no_slip")



    # flow interior low res away from airfoil
    lr_interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=corrugated_airfoil,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.lr_interior,
        criteria=Or(x < flow_box_origin[0], x > (flow_box_origin[0] + flow_box_dim[0]),y < flow_box_origin[1],y > (flow_box_origin[1] + flow_box_dim[1])),
        lambda_weighting={
            "continuity": Symbol("sdf"),
            "momentum_x": Symbol("sdf"),
            "momentum_y": Symbol("sdf"),
            
        },
        bounds=Bounds({x: channel_length_nd, y: channel_width_nd}),
        batch_per_epoch=8000,

        parameterization=time_range,
    )
    
    ic_domain.add_constraint(lr_interior, name="lr_interior")
    window_domain.add_constraint(lr_interior, name="lr_interior")

    # flow interiror high res near airfoil
    hr_interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=corrugated_airfoil,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.hr_interior,
        criteria=And(
            x > flow_box_origin[0], x < (flow_box_origin[0] + flow_box_dim[0]),y > flow_box_origin[1], y < (flow_box_origin[1] + flow_box_dim[1])
        ),
        lambda_weighting={
            "continuity": Symbol("sdf"),
            "momentum_x": Symbol("sdf"),
            "momentum_y": Symbol("sdf"),
            
        },
        bounds=Bounds({x: channel_length_nd, y: channel_width_nd}),
        batch_per_epoch=16000,

        parameterization=time_range,
        
    )
    
    ic_domain.add_constraint(hr_interior, name="hr_interior")
    window_domain.add_constraint(hr_interior, name="hr_interior")

    # make solver
    slv = SequentialSolver(
        cfg,
        [(1, ic_domain), (nr_time_windows, window_domain)],
        custom_update_operation=time_window_net.move_window,
    )

    # start solver
    slv.solve()

if __name__ == "__main__":
    run()
