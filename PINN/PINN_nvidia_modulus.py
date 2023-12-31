


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

def rotate_point(point, angle_deg):
    # Convert angle from degrees to radians
    angle_rad = np.deg2rad(angle_deg)

    # Define the rotation matrix
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])

    # Apply the rotation matrix to the point
    rotated_point = np.dot(rotation_matrix, point)

    return rotated_point

def boundaryNACA4D(M, P, SS, c, n, offset_x, offset_y,angle_deg):
    """
    Compute the coordinates of a NACA 4-digits airfoil

    Args:
        M:  maximum camber value (*100)
        P:  position of the maximum camber alog the chord (*10)
        SS: maximum thickness (*100)
        c:  chord length
        n:  the total points sampled will be 2*n
    """
    m = M / 100
    p = P / 10
    t = SS / 100

    if (m == 0):
        p = 1

    # Chord discretization (cosine discretization)
    xv = np.linspace(0.0, c, n+1)
    xv = c / 2.0 * (1.0 - np.cos(np.pi * xv / c))

    # Thickness distribution
    ytfcn = lambda x: 5 * t * c * (0.2969 * (x / c)**0.5 -
                                   0.1260 * (x / c) -
                                   0.3516 * (x / c)**2 +
                                   0.2843 * (x / c)**3 -
                                   0.1015 * (x / c)**4)
    yt = ytfcn(xv)

    # Camber line
    yc = np.zeros(np.size(xv))

    for ii in range(n+1):
        if xv[ii] <= p * c:
            yc[ii] = c * (m / p**2 * (xv[ii] / c) * (2 * p - (xv[ii] / c)))
        else:
            yc[ii] = c * (m / (1 - p)**2 * (1 + (2 * p - (xv[ii] / c)) * (xv[ii] / c) - 2 * p))

    # Camber line slope
    dyc = np.zeros(np.size(xv))

    for ii in range(n+1):
        if xv[ii] <= p * c:
            dyc[ii] = m / p**2 * 2 * (p - xv[ii] / c)
        else:
            dyc[ii] = m / (1 - p)**2 * 2 * (p - xv[ii] / c)

    # Boundary coordinates and sorting
    th = np.arctan2(dyc, 1)
    xU = xv - yt * np.sin(th)
    yU = yc + yt * np.cos(th)
    xL = xv + yt * np.sin(th)
    yL = yc - yt * np.cos(th)

    x = np.zeros(2 * n + 1)
    y = np.zeros(2 * n + 1)

    for ii in range(n):
        x[ii] = xL[n - ii]
        y[ii] = yL[n - ii]

    x[n : 2 * n + 1] = xU
    y[n : 2 * n + 1] = yU

    rotated_airfoil = [rotate_point([x, y], angle_deg) for x, y in zip(x, y)]

    return np.array(rotated_airfoil) + np.array([offset_x, offset_y])

boundaryNACA4D(0, 0, 12, 0.2, 250, 0.20, 0.35,-10)

M = 0
P = 0
SS = 12
c = 0.12
n = 250
offset_x = 0.20
offset_y = 0.35
angle_deg = -10

    # make naca geometry
x = [x for x in np.linspace(0, 0.2, 10)] + [x for x in np.linspace(0.2, 1.0, 10)][
    1:
]  # higher res in front
line = boundaryNACA4D(M, P, SS, c, n, offset_x, offset_y,angle_deg)[:-1]
geo = Polygon(line)

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
    airfoil = geo()
    airfoil_geom = channel - airfoil

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