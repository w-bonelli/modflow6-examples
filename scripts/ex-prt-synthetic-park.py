# ## Synthetic Park Particle Tracking Problem
#
# Application of a MODFLOW 6 particle-tracking (PRT)
# model and a MODPATH 7 (MP7) model to model solute
# transport in a synthetic flow system simulating a
# public park with a stream, a drain, a well, and a
# river in a hilly landscape and steady-state flow.
#
# Particles are released in two groups, one at each
# lower corner of the grid.
#

# ### Initial setup
#
# Import dependencies, define the example name and workspace,
# and read settings from environment variables.

# +
import pathlib as pl
from pprint import pformat

import flopy
import git
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flopy.mf6 import MFSimulation
from flopy.plot.styles import styles
from flopy.utils.gridintersect import GridIntersect
from matplotlib import colormaps as cm
from modflow_devtools.misc import get_env, timed
from shapely.geometry import LineString, Point, Polygon

# Example name and workspace paths. If this example is running
# in the git repository, use the folder structure described in
# the README. Otherwise just use the current working directory.
sim_name = "ex-prt-syn-prk"
# shorten model names so they fit in 16-char limit
gwf_name = sim_name.replace("ex-prt-", "") + "-gwf"
prt_name = sim_name.replace("ex-prt-", "") + "-prt"
mp7_name = sim_name.replace("ex-prt-", "") + "-mp7"
try:
    root = pl.Path(git.Repo(".", search_parent_directories=True).working_dir)
except:
    root = None
workspace = root / "examples" if root else pl.Path.cwd()
figs_path = root / "figures" if root else pl.Path.cwd()
sim_ws = workspace / sim_name
gwf_ws = sim_ws / "gwf"
prt_ws = sim_ws / "prt"
mp7_ws = sim_ws / "mp7"
gwf_ws.mkdir(exist_ok=True, parents=True)
prt_ws.mkdir(exist_ok=True, parents=True)
mp7_ws.mkdir(exist_ok=True, parents=True)

# Define output file names
headfile = f"{gwf_name}.hds"
budgetfile = f"{gwf_name}.cbb"
budgetfile_prt = f"{prt_name}.cbb"
trackfile_prt = f"{prt_name}.trk"
trackhdrfile_prt = f"{prt_name}.trk.hdr"
trackcsvfile_prt = f"{prt_name}.trk.csv"
pathlinefile_mp7 = f"{mp7_name}.mppth"
endpointfile_mp7 = f"{mp7_name}.mpend"

# Settings from environment variables
write = get_env("WRITE", True)
run = get_env("RUN", True)
plot = get_env("PLOT", True)
plot_show = get_env("PLOT_SHOW", True)
plot_save = get_env("PLOT_SAVE", True)
# -

# ### Define parameters
#
# Define model units, parameters and other settings.

# +
# Model units
length_units = "feet"
time_units = "days"

# Model parameters
nper = 1  # Number of periods
nlay = 3  # Number of layers
delr = 25.0  # Column width ($ft$)
delc = 25.0  # Row width ($ft$)
botm_str = "950.0, 940.0, 800.0"  # Layer bottom elevations ($ft$)
kh_str = "50.0, 10.0, 100.0"  # Horizontal hydraulic conductivity ($ft/d$)
kv_str = "10.0, 5.0, 20.0"  # Vertical hydraulic conductivity ($ft/d$)
rchv = 0.005  # Recharge rate ($ft/d$)
str_h = 980.0  # Stream stage ($ft$)
str_z = 960.0  # Stream bottom ($ft$)
str_c = 1.0e5  # Stream conductance ($ft^2/d$)
riv_h = 980.0  # River stage ($ft$)
riv_z = 960.0  # River bottom ($ft$)
riv_c = 1.0e5  # River conductance ($ft^2/d$)
porosity = 0.1  # Soil porosity (unitless)

# Time discretization
perioddata = [
    # perlen, nstp, tsmult
    (100000, 1, 1),
]

# Parse bottom elevation and horiz/vert hydraulic cond.
botm = [float(value) for value in botm_str.split(",")]
kh = [float(value) for value in kh_str.split(",")]
kv = [float(value) for value in kv_str.split(",")]

# Define top elevations
top = 990

laytyp = [1, 0, 0]

# Define well data.
# Negative discharge indicates pumping, positive injection.
wells = [
    # layer, row, col, discharge
    (0, 237, -705000),
]

# Define the river location.
riv_iface = 6
riv_iflowface = -1
riv_cells = [(0, 120)] + [(0, j) for j in range(35, 66)] + [(0, 124)]
rd = [[(k, j), riv_h, riv_c, riv_z, riv_iface, riv_iflowface] for k, j in riv_cells]

# Define hill contours.
nw_hill_contours = [
    [(70, 690), (30, 530), (210, 530), (270, 490)],
]
sw_hill_contours = [
    [(10, 110), (70, 110), (70, 10)],
    [(10, 150), (90, 190), (130, 190), (190, 170), (170, 110), (150, 10)],
    [(10, 190), (110, 230), (170, 230), (210, 210), (230, 110), (210, 10)],
    [(10, 230), (50, 250), (110, 270), (230, 230), (270, 190), (270, 130), (250, 10)],
    [(10, 270), (110, 290), (250, 270), (290, 250), (310, 30)],
]
se_hill_contours = [
    [
        (650, 10),
        (890, 510),
    ],
    [
        (710, 10),
        (890, 450),
    ],
    [
        (790, 10),
        (890, 390),
    ],
]

# Define river line.
river_line = [list(zip(range(590, 891, 10), range(10, 651, 19)))]

# -

# ### Grid creation and refinement
#
# [GRIDGEN](https://www.usgs.gov/software/gridgen-program-generating-unstructured-finite-volume-grids) can be used to create a quadpatch grid with a central refined region.
#
# The grid will have several refinement features. First, create the top-level (base) grid discretization.


def get_disvprops():
    from flopy.discretization import VertexGrid
    from flopy.utils.triangle import Triangle as Triangle
    from flopy.utils.voronoi import VoronoiGrid

    domain = [(0, 0), (925, 0), (925, 700), (0, 700)]
    tri = Triangle(angle=30, model_ws=workspace)

    # create active domain first
    tri.add_polygon(domain)

    # create refinement features:
    #   - hills in northwest, southwest, southeast
    #   - river running diagonal on the right side
    for poly in sw_hill_contours + nw_hill_contours:
        tri.add_polygon(poly)

    for poly in river_line:
        tri.add_polygon(poly)

    tri.build(verbose=False)

    vor = VoronoiGrid(tri)
    gridprops = vor.get_gridprops_vertexgrid()
    voronoi_grid = VertexGrid(**gridprops, nlay=1)

    return voronoi_grid, vor.get_disv_gridprops()


voronoi_grid, disvkwargs = get_disvprops()
gi = GridIntersect(voronoi_grid, method="vertex")
topa = np.ones((voronoi_grid.ncpl,)) * top
# southwest hill
sw_hc3_cells = gi.intersect(Polygon(sw_hill_contours[3] + [(0, 0)])).cellids.astype(int)
sw_hc2_cells = gi.intersect(Polygon(sw_hill_contours[2] + [(0, 0)])).cellids.astype(int)
sw_hc1_cells = gi.intersect(Polygon(sw_hill_contours[1] + [(0, 0)])).cellids.astype(int)
sw_hc0_cells = gi.intersect(Polygon(sw_hill_contours[0] + [(0, 0)])).cellids.astype(int)
topa[sw_hc3_cells] = 1010
topa[sw_hc2_cells] = 1030
topa[sw_hc1_cells] = 1050
topa[sw_hc0_cells] = 1070
# northwest hill
nw_hc0_cells = gi.intersect(LineString(nw_hill_contours[0])).cellids.astype(int)
topa[nw_hc0_cells] = 1010
# southeast hill
topa[gi.intersect(LineString(se_hill_contours[0])).cellids.astype(int)] = 1000
topa[gi.intersect(LineString(se_hill_contours[1])).cellids.astype(int)] = 1010
topa[gi.intersect(LineString(se_hill_contours[2])).cellids.astype(int)] = 1020

# Get well and river cell numbers
nodes = {"well": [], "river": []}
for k, j, _ in wells:
    nn = k * voronoi_grid.ncpl + j
    nodes["well"].append(nn)
for rivspec in rd:
    k, j = rivspec[0]
    nn = k * voronoi_grid.ncpl + j
    nodes["river"].append(nn)


# -

# ### Model setup
#
# Define functions to build models, write input files, and run the simulation.


# +


def build_gwf_model():
    # simulation
    sim = flopy.mf6.MFSimulation(
        sim_name=sim_name, exe_name="mf6", version="mf6", sim_ws=gwf_ws
    )

    # temporal discretization
    tdis = flopy.mf6.modflow.mftdis.ModflowTdis(
        sim,
        pname="tdis",
        time_units=time_units,
        nper=len(perioddata),
        perioddata=perioddata,
    )

    # groundwater flow (gwf) model
    model_nam_file = f"{gwf_name}.nam"
    gwf = flopy.mf6.ModflowGwf(
        sim, modelname=gwf_name, model_nam_file=model_nam_file, save_flows=True
    )

    # iterative model solver (ims) package
    ims = flopy.mf6.modflow.mfims.ModflowIms(
        sim,
        pname="ims",
        complexity="SIMPLE",
    )
    sim.register_solution_package(ims, [gwf.name])

    # grid discretization
    dis = flopy.mf6.modflow.mfgwfdisv.ModflowGwfdisv(
        gwf, nlay=3, top=topa, botm=botm, **disvkwargs
    )

    # initial conditions
    strt = np.ones((voronoi_grid.ncpl,)) * 995
    strt[sw_hc3_cells] = topa[sw_hc3_cells] + (
        topa[sw_hc3_cells] - strt[sw_hc3_cells] / 2
    )
    strt[sw_hc2_cells] = topa[sw_hc2_cells] + (
        topa[sw_hc2_cells] - strt[sw_hc2_cells] / 2
    )
    strt[sw_hc1_cells] = topa[sw_hc1_cells] + (
        topa[sw_hc1_cells] - strt[sw_hc1_cells] / 2
    )
    strt[sw_hc0_cells] = topa[sw_hc0_cells] + (
        topa[sw_hc0_cells] - strt[sw_hc0_cells] / 2
    )
    strt = np.stack([strt, strt, strt])
    ic = flopy.mf6.modflow.mfgwfic.ModflowGwfic(gwf, pname="ic", strt=strt)

    # node property flow
    npf = flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf(
        gwf,
        pname="npf",
        # icelltype=laytyp,
        k=kh,
        k33=kv,
        save_flows=True,
        save_specific_discharge=True,
        save_saturation=True,
    )

    # constant head boundary on right side
    cells_left = gi.intersect(LineString([(0, 0), (0, 700)]))["cellids"]
    cells_right = gi.intersect(LineString([(924, 0), (924, 700)]))["cellids"]
    chdlist = []
    for i, ic in enumerate(cells_left):
        chdlist.append([(0, ic), 1010.0 + i])
    for i, ic in enumerate(cells_right):
        chdlist.append([(0, ic), 980.0 + i])
    chd = flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chdlist)

    # recharge
    rch = flopy.mf6.modflow.mfgwfrcha.ModflowGwfrcha(gwf, recharge=rchv)

    # storage
    sto = flopy.mf6.modflow.mfgwfsto.ModflowGwfsto(
        gwf,
        save_flows=True,
        iconvert=1,
        ss=0.0001,
        sy=0.1,
        steady_state={0: True},
    )

    # wells
    wel = flopy.mf6.modflow.mfgwfwel.ModflowGwfwel(
        gwf,
        maxbound=1,
        stress_period_data={0: wells},
    )

    # river
    flopy.mf6.modflow.mfgwfriv.ModflowGwfriv(
        gwf, auxiliary=["iface", "iflowface"], stress_period_data={0: rd}
    )

    # output control
    oc = flopy.mf6.modflow.mfgwfoc.ModflowGwfoc(
        gwf,
        pname="oc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        head_filerecord=[headfile],
        budget_filerecord=[budgetfile],
    )

    return sim


def build_prt_model():
    # simulation
    sim = flopy.mf6.MFSimulation(
        sim_name=sim_name, exe_name="mf6", version="mf6", sim_ws=prt_ws
    )

    # temporal discretization
    tdis = flopy.mf6.modflow.mftdis.ModflowTdis(
        sim,
        pname="tdis",
        time_units=time_units,
        nper=len(perioddata),
        perioddata=perioddata,
    )

    # Instantiate the MODFLOW 6 prt model
    prt = flopy.mf6.ModflowPrt(
        sim, modelname=prt_name, model_nam_file="{}.nam".format(prt_name)
    )

    # Instantiate the MODFLOW 6 prt discretization package
    voronoi_grid, disvkwargs = get_disvprops()
    gi = GridIntersect(voronoi_grid, method="vertex")
    topa = np.ones((voronoi_grid.ncpl,)) * top
    gi = GridIntersect(voronoi_grid, method="vertex")
    topa = np.ones((voronoi_grid.ncpl,)) * top
    # southwest hill
    sw_hc3_cells = gi.intersect(Polygon(sw_hill_contours[3] + [(0, 0)])).cellids.astype(
        int
    )
    sw_hc2_cells = gi.intersect(Polygon(sw_hill_contours[2] + [(0, 0)])).cellids.astype(
        int
    )
    sw_hc1_cells = gi.intersect(Polygon(sw_hill_contours[1] + [(0, 0)])).cellids.astype(
        int
    )
    sw_hc0_cells = gi.intersect(Polygon(sw_hill_contours[0] + [(0, 0)])).cellids.astype(
        int
    )
    topa[sw_hc3_cells] = 1010
    topa[sw_hc2_cells] = 1030
    topa[sw_hc1_cells] = 1050
    topa[sw_hc0_cells] = 1070
    # northwest hill
    nw_hc0_cells = gi.intersect(LineString(nw_hill_contours[0])).cellids.astype(int)
    topa[nw_hc0_cells] = 1010
    # southeast hill
    topa[gi.intersect(LineString(se_hill_contours[0])).cellids.astype(int)] = 1000
    topa[gi.intersect(LineString(se_hill_contours[1])).cellids.astype(int)] = 1010
    topa[gi.intersect(LineString(se_hill_contours[2])).cellids.astype(int)] = 1020
    dis = flopy.mf6.modflow.mfgwfdisv.ModflowGwfdisv(
        prt, nlay=3, top=topa, botm=botm, **disvkwargs
    )

    # Instantiate the MODFLOW 6 prt model input package.
    flopy.mf6.ModflowPrtmip(prt, pname="mip", porosity=porosity)

    # Convert MODPATH 7 particle configuration to format expected by PRP.
    #     points = list(zip(, range(25, 50)))
    points = np.transpose(
        np.array(np.meshgrid(range(25, 50), range(25, 50))).reshape(2, -1)
    )

    def intersect(p):
        return gi.intersect(Point(p))[0].cellids

    release_points = [
        [i, (0, intersect(p)), p[0], p[1], 1020.0] for i, p in enumerate(points)
    ]

    # Instantiate the MODFLOW 6 PRT particle release point (PRP) package.
    flopy.mf6.ModflowPrtprp(
        prt,
        pname="prp1",
        filename="{}_1.prp".format(prt_name),
        nreleasepts=len(release_points),
        packagedata=release_points,
        # local z coordinates specified, compute global release
        # coord from cell top if saturated or water table if not
        # local_z=True,
        exit_solve_tolerance=1e-5,
    )

    # Instantiate the MODFLOW 6 prt output control package
    tracktimes = list(range(0, 200, 2))
    flopy.mf6.ModflowPrtoc(
        prt,
        pname="oc",
        budget_filerecord=[budgetfile_prt],
        trackcsv_filerecord=[trackcsvfile_prt],
        track_timesrecord=tracktimes,
        saverecord=[("BUDGET", "ALL")],
    )

    # Instantiate the MODFLOW 6 prt flow model interface
    flopy.mf6.ModflowPrtfmi(
        prt,
        packagedata=[
            ("GWFHEAD", pl.Path(f"../{gwf_ws.name}/{headfile}")),
            ("GWFBUDGET", pl.Path(f"../{gwf_ws.name}/{budgetfile}")),
        ],
    )

    # Create an explicit model solution (EMS) for the MODFLOW 6 prt model
    ems = flopy.mf6.ModflowEms(
        sim,
        pname="ems",
        filename=f"{prt_name}.ems",
    )
    sim.register_solution_package(ems, [prt.name])

    return sim


def build_models():
    gwfsim = build_gwf_model()
    prtsim = build_prt_model()
    # mp7sim = build_mp7_model(gwfsim.get_model(gwf_name))
    return gwfsim, prtsim  # , mp7sim


def write_models(*sims, silent=False):
    for sim in sims:
        if isinstance(sim, MFSimulation):
            sim.write_simulation(silent=silent)
        else:
            sim.write_input()


@timed
def run_models(*sims, silent=False):
    for sim in sims:
        if isinstance(sim, MFSimulation):
            success, buff = sim.run_simulation(silent=silent, report=True)
        else:
            success, buff = sim.run_model(silent=silent, report=True)
        assert success, pformat(buff)


# -

# ### Plotting results
#
# Define functions to plot model results.


# +
# Pathline and starting point colors by destination
colordest = {"well": "red", "drain": "green", "stream": "blue"}


def plot_head(gwf, head, spdis, pls):
    with styles.USGSPlot():
        fig, ax = plt.subplots(figsize=(7, 7))
        fig.tight_layout()
        ax.set_aspect("equal")
        ilay = 2
        cint = 1
        hmin = head[ilay].min()
        hmax = head[ilay].max()
        styles.heading(ax=ax, heading=f"Head, layer {str(ilay + 1)}, time=0")
        mm = flopy.plot.PlotMapView(gwf, ax=ax, layer=ilay)
        mm.plot_grid(lw=0.5)
        mm.plot_bc("WEL", plotAll=True, color="red", alpha=0.5)
        mm.plot_bc("RIV", plotAll=True, color="blue", alpha=0.5)

        # plot head array and contours
        pc = mm.plot_array(head, edgecolor="black", alpha=0.25, cmap="viridis")
        cb = plt.colorbar(pc, shrink=0.25)
        cb.ax.set_xlabel(r"Head ($ft$)")
        levels = np.arange(np.floor(hmin), np.ceil(hmax) + cint, cint)
        cs = mm.contour_array(head, colors="black", levels=levels, alpha=0.5)
        plt.clabel(cs, fmt="%.1f", colors="black", fontsize=11)

        # plot specific discharge
        mm.plot_vector(*spdis, normalize=True, alpha=0.15)

        # create legend
        ax.legend(
            handles=[
                mpl.patches.Patch(color="red", label="Well", alpha=0.5),
                mpl.patches.Patch(color="blue", label="River", alpha=0.5),
                mpl.lines.Line2D(
                    [0],
                    [0],
                    marker=">",
                    linestyle="",
                    label="Specific discharge",
                    color="grey",
                    markerfacecolor="gray",
                ),
            ],
            loc="upper right",
        )

        pts = ax.scatter(pls["x"], pls["y"], s=2, c=pls["t"], cmap=cm["magma"])
        cb2 = plt.colorbar(pts, shrink=0.25)
        cb2.ax.set_xlabel("Travel time (days)")

        # plot node numbers
        # for i in range(gwf.modelgrid.ncpl):
        #     x, y = gwf.modelgrid.xcellcenters[i], gwf.modelgrid.ycellcenters[i]
        #     color = "grey"
        #     ms = 2
        #     ax.plot(x, y, "o", color=color, alpha=0.25, ms=ms)
        #     ax.annotate(str(i + 1), (x, y), color="grey", alpha=0.9)

        if plot_show:
            plt.show()
        if plot_save:
            fig.savefig(figs_path / f"{sim_name}-head")


def plot_pathpoints_3d(gwf, pls, title=None):
    import pyvista as pv
    from flopy.export.vtk import Vtk

    pv.set_plot_theme("document")
    axes = pv.Axes(show_actor=False, actor_scale=2.0, line_width=5)
    vert_exag = 1
    vtk = Vtk(model=gwf, binary=False, vertical_exageration=vert_exag, smooth=False)
    vtk.add_model(gwf)
    vtk.add_pathline_points(pls)
    gwf_mesh, prt_mesh = vtk.to_pyvista()
    gwf_mesh.rotate_z(110, point=axes.origin, inplace=True)
    gwf_mesh.rotate_y(-10, point=axes.origin, inplace=True)
    gwf_mesh.rotate_x(10, point=axes.origin, inplace=True)
    prt_mesh.rotate_z(110, point=axes.origin, inplace=True)
    prt_mesh.rotate_y(-10, point=axes.origin, inplace=True)
    prt_mesh.rotate_x(10, point=axes.origin, inplace=True)

    # import pdb; pdb.set_trace()
    top_mesh = gwf_mesh.remove_cells(
        list(set(range(voronoi_grid.ncpl, voronoi_grid.ncpl * 3))), inplace=False
    )
    well_mesh = gwf_mesh.remove_cells(
        list(set(range(voronoi_grid.nnodes * 3)) - set(nodes["well"])), inplace=False
    )
    river_mesh = gwf_mesh.remove_cells(
        list(set(range(voronoi_grid.nnodes * 3)) - set(nodes["river"])), inplace=False
    )

    def _plot(screenshot=False):
        p = pv.Plotter(
            window_size=[500, 500],
            off_screen=screenshot,
            notebook=False if screenshot else None,
        )
        p.enable_anti_aliasing()
        if title is not None:
            p.add_title(title, font_size=5)
        p.add_mesh(
            gwf_mesh,
            opacity=0.05,
            # style="wireframe",
        )
        head = gwf.output.head().get_data()
        p.add_mesh(
            top_mesh, scalars=head[2], cmap="viridis", opacity=0.02, show_edges=False
        )
        p.add_mesh(well_mesh, color="red", opacity=0.5, show_edges=False)
        p.add_mesh(river_mesh, color="blue", opacity=0.05, show_edges=False)
        p.add_mesh(
            prt_mesh,
            scalars="t",
            opacity=0.2,
            cmap="magma",
            point_size=3,
            line_width=2,
            render_points_as_spheres=True,
            render_lines_as_tubes=True,
            smooth_shading=True,
        )
        p.remove_scalar_bar("t")
        p.remove_scalar_bar()

        p.camera.zoom(2)
        p.show()
        if screenshot:
            p.screenshot(figs_path / f"{sim_name}-paths-3d.png")

    if plot_show:
        _plot()
    if plot_save:
        _plot(screenshot=True)


def plot_all(gwfsim):
    # load results
    gwf = gwfsim.get_model(gwf_name)
    head = gwf.output.head().get_data()
    bdobj = gwf.output.budget()
    spdis = bdobj.get_data(text="DATA-SPDIS")[0]
    qx, qy, _ = flopy.utils.postprocessing.get_specific_discharge(spdis, gwf)
    pls = pd.read_csv(prt_ws / trackcsvfile_prt, na_filter=False)
    pls = pls[pls.ireason != 1]

    # import pdb; pdb.set_trace()

    # plot the results
    plot_head(gwf, head=head, spdis=(qx, qy), pls=pls)
    plot_pathpoints_3d(gwf, pls=pls)


# -

# ### Running the example
#
# Define a function to run the example scenarios and plot results.


def scenario(silent=False):
    gwfsim, prtsim = build_models()
    if write:
        write_models(gwfsim, prtsim, silent=silent)
    if run:
        run_models(gwfsim, prtsim, silent=silent)
    if plot:
        plot_all(gwfsim)


# Run the MODPATH 7 example problem 3 scenario.

scenario(silent=False)
