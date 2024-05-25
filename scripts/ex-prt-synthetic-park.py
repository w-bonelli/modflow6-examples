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
from matplotlib.lines import Line2D
from modflow_devtools.misc import get_env, timed

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
nrow = 28  # Number of rows
ncol = 37  # Number of columns
delr = 20.0  # Column width ($ft$)
delc = 20.0  # Row width ($ft$)
botm_str = "950.0, 920.0, 800.0"  # Layer bottom elevations ($ft$)
kh_str = "50.0, 10.0, 100.0"  # Horizontal hydraulic conductivity ($ft/d$)
kv_str = "10.0, 10.0, 20.0"  # Vertical hydraulic conductivity ($ft/d$)
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
top = np.ones((28, 37)) * 990
top[0:20,0:20] = 1000.
top[0:19,0:19] = 1010.
top[0:18,0:18] = 1020.
top[0:17,0:17] = 1030.
top[0:16,0:16] = 1040.
top[0:15,0:15] = 1050.
top[0:14,0:14] = 1060.
top[0:13,0:13] = 1070.
top[0:12,0:12] = 1080.
top[0:11,0:11] = 1090.
top[0:10,0:10] = 1100.
top[0:9,0:9] = 1110.

laytyp = [1, 0, 0]

# Define well data.
# Negative discharge indicates pumping, positive injection.
wells = [
    # layer, row, col, discharge
    (0, 1, 5, -5000),
]

# Define the drain location.
drain = (0, 5, (23, 36))

# Define the stream location.
str_iface = 6
str_iflowface = -1
str1_cells = [
    (0, 1, 6), 
    (0, 1, 7),
    (0, 1, 8),
    (0, 2, 8),
    (0, 3, 9),
    (0, 3, 10),
    (0, 3, 11),
    (0, 4, 12),
    (0, 4, 13),
    (0, 4, 14),
    (0, 4, 15)
]
sd = []
for cell in str1_cells:
    k, i, j = cell
    sd.append([(k, i, j), str_h, str_c, str_z, str_iface, str_iflowface])

# Define the river location.
riv_iface = 6
riv_iflowface = -1
rd = []
for i in range(nrow):
    rd.append([(0, i, ncol - 1), riv_h, riv_c, riv_z, riv_iface, riv_iflowface])

# -

# ### Grid refinement
#
# [GRIDGEN](https://www.usgs.gov/software/gridgen-program-generating-unstructured-finite-volume-grids) can be used to create a quadpatch grid with a central refined region.
#
# The grid will have several refinement features. First, create the top-level (base) grid discretization.




# Configure locations for particle tracking to terminate. We
# have three explicitly defined termination zones:
#
# - 2: the well in layer 1, at row 1, column 6
# - 3: the drain in layer 1, running through row 6 from column 24-36
# - 4: the stream in layer 1, in the upper left of the grid
#
# MODFLOW 6 reserves zone number 1 to indicate that particles
# may move freely within the zone.

# +
def get_izone():
    izone = []

    # zone 1 is the default (non-terminating)
    def ones():
        return np.ones((nrow, ncol), dtype=np.int32)

    # layer 1
    l1 = ones()
    l1[wells[0][1:3]] = 2  # well
    l1[drain[1], drain[2][0] : drain[2][1]] = 3  # drain
    for cell in sd:
        l1[cell[0][1:3]] = 4  # stream
    l1[:, ncol - 1] = 5  # river
    izone.append(l1)

    return izone

izone = get_izone()

# -

#
# Define particles to track. Particles are released from the top of a
# 2x2 square of cells in the upper left of the midel grid's top layer.
# MODPATH 7 uses a reference time value of 0.9 to start the release at
# 90,000 days into the simulation.

# +

rel_minl = rel_maxl = 0
rel_minr = 23
rel_maxr = 28
rel_minc = 0
rel_maxc = 6
celldata = flopy.modpath.CellDataType(
    drape=0,
    rowcelldivisions=5,
    columncelldivisions=5,
    layercelldivisions=1,
)
lrcregions = [[rel_minl, rel_minr, rel_minc, rel_maxl, rel_maxr, rel_maxc]]
lrcpd = flopy.modpath.LRCParticleData(
    subdivisiondata=[celldata],
    lrcregions=[lrcregions],
)
pg = flopy.modpath.ParticleGroupLRCTemplate(
    particlegroupname="PG1",
    particledata=lrcpd,
    filename=f"{mp7_name}.pg1.sloc",
)
pgs = [pg]
defaultiface = {"RECHARGE": 6, "ET": 6}

# Define well and river cell numbers, used to extract and plot model results later.

# Get well and river cell numbers
nodes = {"well": [], "drain": [], "stream": [], "river": []}
for k, i, j, _ in wells:
    nodes[f"well"].append(ncol * (nrow * k + i) + j)
for j in drain[2]:
    k, i = drain[:2]
    nodes["drain"].append([ncol * (nrow * k + i) + j])
for strspec in sd:
    k, i, j = strspec[0]
    node = ncol * (nrow * k + i) + j
    nodes["stream"].append(node)
for rivspec in rd:
    k, i, j = rivspec[0]
    node = ncol * (nrow * k + i) + j
    nodes["river"].append(node)
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

    ms = flopy.modflow.Modflow()
    dis = flopy.modflow.ModflowDis(
        ms,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )

    # grid discretization
    dis = flopy.mf6.modflow.mfgwfdis.ModflowGwfdis(
        gwf,
        pname="dis",
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        length_units="FEET",
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )

    # initial conditions
    strt = np.ones((28, 37)) * 990.
    strt[0:20,0:20] = 995.
    strt[0:19,0:19] = 1000.
    strt[0:18,0:18] = 1010.
    strt[0:17,0:17] = 1020.
    strt[0:16,0:16] = 1030.
    strt[0:15,0:15] = 1040.
    strt[0:14,0:14] = 1050.
    strt[0:13,0:13] = 1060.
    strt[0:12,0:12] = 1070.
    strt[0:11,0:11] = 1080.
    strt[0:10,0:10] = 1090.
    strt[0:9,0:9] = 1100.
    strt = np.stack([strt, strt, strt])
    ic = flopy.mf6.modflow.mfgwfic.ModflowGwfic(gwf, pname="ic", strt=strt)

    # node property flow
    npf = flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf(
        gwf,
        pname="npf",
        icelltype=laytyp,
        k=kh,
        k33=kv,
        save_flows=True,
        save_specific_discharge=True,
        save_saturation=True,
    )

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

    # streams
    flopy.mf6.modflow.mfgwfriv.ModflowGwfriv(
        gwf, auxiliary=["iface", "iflowface"], stress_period_data={0: sd + rd}
    )

    # drain (set auxiliary IFACE var to 6 for top of cell)
    drn_iface = 6
    drn_iflowface = -1
    dd = [
        [drain[0], drain[1], i + drain[2][0], 990., 100000.0, drn_iface, drn_iflowface]
        for i in range(drain[2][1] - drain[2][0])
    ]
    drn = flopy.mf6.modflow.mfgwfdrn.ModflowGwfdrn(
        gwf, auxiliary=["iface", "iflowface"], maxbound=13, stress_period_data={0: dd}
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


def build_models():
    gwfsim = build_gwf_model()
    # prtsim = build_prt_model()
    # mp7sim = build_mp7_model(gwfsim.get_model(gwf_name))
    return gwfsim # , prtsim, mp7sim


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
        # assert success, pformat(buff)

# -

# ### Plotting results
#
# Define functions to plot model results.


# +
# Pathline and starting point colors by destination
colordest = {"well": "red", "drain": "green", "stream": "blue"}


def plot_head(gwf, head):
    with styles.USGSPlot():
        fig, ax = plt.subplots(figsize=(7, 7))
        fig.tight_layout()
        ax.set_aspect("equal")
        ilay = 2
        cint = 0.25
        hmin = head[ilay, 0, :].min()
        hmax = head[ilay, 0, :].max()
        styles.heading(ax=ax, heading=f"Head, layer {str(ilay + 1)}, time=0")
        mm = flopy.plot.PlotMapView(gwf, ax=ax, layer=ilay)
        mm.plot_grid(lw=0.5)
        mm.plot_bc("WEL", plotAll=True, color="red", alpha=0.5)
        mm.plot_bc("RIV", plotAll=True, color="blue", alpha=0.5)
        mm.plot_bc("DRN", plotAll=True, color="green", alpha=0.5)

        pc = mm.plot_array(head[ilay, :, :], edgecolor="black", alpha=0.1)
        cb = plt.colorbar(pc, shrink=0.25, pad=0.1)
        cb.ax.set_xlabel(r"Head ($ft$)")

        levels = np.arange(np.floor(hmin), np.ceil(hmax) + cint, cint)
        cs = mm.contour_array(head[ilay, :, :], colors="white", levels=levels)
        plt.clabel(cs, fmt="%.1f", colors="white", fontsize=11)

        ax.legend(
            handles=[
                mpl.patches.Patch(color="red", label="Well", alpha=0.5),
                mpl.patches.Patch(color="blue", label="Stream/River", alpha=0.5),
                mpl.patches.Patch(color="green", label="Drain", alpha=0.5),
            ],
            loc="upper right",
        )

        if plot_show:
            plt.show()
        if plot_save:
            fig.savefig(figs_path / f"{sim_name}-head")


def plot_pathpoints_3d(gwf, title=None):
    import pyvista as pv
    from flopy.export.vtk import Vtk

    pv.set_plot_theme("document")
    axes = pv.Axes(show_actor=False, actor_scale=2.0, line_width=5)
    vert_exag = 1
    vtk = Vtk(model=gwf, binary=False, vertical_exageration=vert_exag, smooth=False)
    vtk.add_model(gwf)
    # vtk.add_pathline_points(mf6pl)
    gwf_mesh = vtk.to_pyvista()
    gwf_mesh.rotate_z(110, point=axes.origin, inplace=True)
    gwf_mesh.rotate_y(-10, point=axes.origin, inplace=True)
    gwf_mesh.rotate_x(10, point=axes.origin, inplace=True)

    def _plot(screenshot=False):
        p = pv.Plotter(
            window_size=[500, 500],
            off_screen=screenshot,
            notebook=False if screenshot else None,
        )
        p.enable_anti_aliasing()
        if title is not None:
            p.add_title(title, font_size=5)
        p.add_mesh(gwf_mesh, opacity=0.25)
        p.add_legend(
            labels=[
                ("Well (layer 3)", "red"),
                ("Drain", "green"),
                ("River", "blue"),
            ],
            bcolor="white",
            face="r",
            size=(0.15, 0.15),
        )

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
    head = flopy.utils.HeadFile(gwf_ws / (gwf_name + ".hds")).get_data()

    # plot the results
    plot_head(gwf, head=head)
    plot_pathpoints_3d(gwf)

# -

# ### Running the example
#
# Define a function to run the example scenarios and plot results.


def scenario(silent=False):
    gwfsim = build_models() # , prtsim, mp7sim = build_models()
    if write:
        write_models(gwfsim, silent=silent)
    if run:
        run_models(gwfsim, silent=silent)
    if plot:
        plot_all(gwfsim)


# Run the MODPATH 7 example problem 3 scenario.

scenario(silent=False)