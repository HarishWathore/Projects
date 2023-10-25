import os
import warnings

from sympy import Symbol, Eq, Abs

import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
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
from sympy import Number, Symbol, Heaviside, atan, sin, cos, sqrt
import os

from modulus.sym.geometry.primitives_2d import Polygon
from modulus.sym.geometry.parameterization import Parameterization, Parameter
from modulus.sym.utils.io.vtk import var_to_polyvtk
from modulus.sym.geometry import Bounds
import modulus.sym
from modulus.sym.hydra import to_yaml
from modulus.sym.hydra.utils import compose
from modulus.sym.hydra.config import ModulusConfig
from modulus.sym.eq.pdes.basic import NormalDotVec
from modulus.sym.eq.non_dim import NonDimensionalizer, Scaler
import naca_airfoil


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    # physical quantities
    nu = quantity(0.02, "kg/(m*s)")
    rho = quantity(1.0, "kg/m^3")
    inlet_u = quantity(1.0, "m/s")
    inlet_v = quantity(0.0, "m/s")
    noslip_u = quantity(0.0, "m/s")
    noslip_v = quantity(0.0, "m/s")
    outlet_p = quantity(0.0, "pa")
    velocity_scale = inlet_u
    density_scale = rho
    length_scale = quantity(20, "m")
    nd = NonDimensionalizer(
        length_scale=length_scale,
        time_scale=length_scale / velocity_scale,
        mass_scale=density_scale * (length_scale**3),
    )

    # geometry
    channel_length = (quantity(-10, "m"), quantity(30, "m"))
    channel_width = (quantity(-10, "m"), quantity(10, "m"))

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
    # Set NACA parameters
    m = 0.02
    p = 0.4
    t = 0.12
    c = 1.0

    # Generate NACA airfoil
    x = [x for x in np.linspace(0, 0.2, 10)] + [x for x in np.linspace(0.2, 1.0, 10)][1:]
    line = naca_airfoil.naca4(x, m, p, t, c)[:-1]

    geo = Polygon(line)
    geometry = channel - geo

    # make list of nodes to unroll graph on
    ns = NavierStokes(nu=nd.ndim(nu), rho=nd.ndim(rho), dim=2, time=False)
    normal_dot_vel = NormalDotVec(["u", "v"])
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = (
        ns.make_nodes()
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")]
        + Scaler(
            ["u", "v", "p"],
            ["u_scaled", "v_scaled", "p_scaled"],
            ["m/s", "m/s", "m^2/s^2"],
            nd,
        ).make_node()
    )

    # make domain
    domain = Domain()
    x, y = Symbol("x"), Symbol("y")

    # inlet
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet,
        outvar={"u": nd.ndim(inlet_u), "v": nd.ndim(inlet_v)},
        batch_size=cfg.batch_size.inlet,
    )
    domain.add_constraint(inlet, "inlet")

    # outlet
    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet,
        outvar={"p": nd.ndim(outlet_p)},
        batch_size=cfg.batch_size.outlet,
    )
    domain.add_constraint(outlet, "outlet")

    # full slip (channel walls)
    walls = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=channel,
        outvar={"u": nd.ndim(inlet_u), "v": nd.ndim(inlet_v)},
        batch_size=cfg.batch_size.walls,
    )
    domain.add_constraint(walls, "walls")

    # no slip
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=airfoil,
        outvar={"u": nd.ndim(noslip_u), "v": nd.ndim(noslip_v)},
        batch_size=cfg.batch_size.no_slip,
    )
    domain.add_constraint(no_slip, "no_slip")

    # interior contraints
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=airfoil_geom,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.interior,
        bounds=Bounds({x: channel_length_nd, y: channel_width_nd}),
    )
    domain.add_constraint(interior, "interior")


    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()

if __name__ == "__main__":
    run()
