'''
The UnbiasedSimulation class for running MD simulations.
'''

import os
from sys import stdout
from pathlib import Path

from mdtraj.reporters import XTCReporter
from .reporter import CVReporter, MinimizationReporter
from . import Charmm36mFF

import openmm
import openmm.app as app
from openmm.unit import angstroms, picoseconds, kelvin, bar
from openmm.unit import Quantity, is_quantity


class UnbiasedSimulation():
    '''
    UnbiasedSimulation class for running MD. An example use is:

    .. code-block:: python

        import af2rave.simulation as af2sim
        sim = af2sim.UnbiasedSimulation(<arguments>)
        sim.run(steps)

    These are the parameters the initialization function accepts:

    :param pdb_file: Path to OpenMM.app.pdbfile.PDBFile object
    :type pdb_file: str
    :param list_of_index: List of indexes to calculate the CVs
    :type list_of_index: list[tuple[int, int]]
    :param forcefield: OpenMM.app.ForceField object. Default: Charmm36mFF
    :type forcefield: OpenMM.app.ForceField
    :param temp: Temperature of the system. Default: 310 K
    :type temp: float | openmm.unit.Quantity[unit=kelvin]
    :param pressure: Pressure of the system. Default: 1 Bar.
        Can be set to none to disable pressure coupling (NVT).
    :type pressure: float | openmm.unit.Quantity[unit=bar]
    :param dt: Time step of the integrator. Default: 0.002 ps
    :type dt: float | openmm.unit.Quantity[unit=picoseconds]
    :param cutoff: Nonbonded cutoff. Default: 10.0 Angstroms
    :type cutoff: float | openmm.unit.Quantity[unit=angstroms]
    :param steps: Simulation steps. Default:  50 million steps (100 ns)
    :type steps: int
    :param cv_file: File to write CVs to. Default: COLVAR.dat
    :type cv_file: str
    :param cv_freq: Frequency of CVs written to the file. Default: 50
    :type cv_freq: int
    :param xtc_file: Trajectory file name. Default: traj.xtc
    :type xtc_file: str
    :param xtc_freq: Frequency writing trajectory in steps. Default: 1000
    :type xtc_freq: int
    :param xtc_reporter: XTCReporter object.
        This overrides traj_filename and traj_freq. Default: None
    :type xtc_reporter: mdtraj.reporters.XTCReporter
    :param append: Appends to existing file. Default: False
    :type append: bool
    :param progress_every: Progress report to stdout frequency. Default: 1000
        Can be set to None to suppress this.
    :type progress_every: int
    '''

    def __init__(self, pdb_file, **kwargs):

        self._prefix = f"{Path(pdb_file).stem}"

        pdb_file = app.pdbfile.PDBFile(pdb_file)
        self._pos = pdb_file.positions
        self._top = pdb_file.topology

        self._pressure = self._get_pressure(**kwargs)
        self._temp = self._get_temperature(**kwargs)

        self._forcefield = kwargs.get('forcefield', Charmm36mFF)
        self._platform = self._choose_platform()
        self.simulation = self._initialize_simulation(**kwargs)

        self._append = kwargs.get('append', False)

        # Reporters
        self._progress_every = kwargs.get('progress_every', 1000)
        self._add_reporter(self._get_cv_reporter(**kwargs))
        self._add_reporter(self._get_xtc_reporter(**kwargs))

        # pressure coupling if needed
        if self._pressure is not None:
            self.simulation.context.getSystem().addForce(
                openmm.MonteCarloBarostat(self._pressure, self._temp)
            )
            self.simulation.context.reinitialize()

    @property
    def pos(self) -> list[openmm.Vec3]:
        '''
        Get the positions of the simulation.

        :return: List of positions
        :rtype: list[openmm.Vec3]
        '''

        self._pos = self.simulation.context.getState(getPositions=True).getPositions()
        return self._pos

    def run(self, steps: int = 50000000) -> app.Simulation:
        '''
        Run the simulation from given pdb file. Default: 50 million steps (100 ns).

        :param steps: Number of steps to run the simulation.
            Default: 50 million steps (100 ns)
        :type steps: int
        '''

        # we have to do this at this later stage, because by design we
        # do not know the number of steps at the time of initialization
        self._add_reporter(self._get_thermo_reporter(steps))
        self.simulation.context.setPositions(self._pos)

        self.simulation.minimizeEnergy(
            maxIterations=500, reporter=MinimizationReporter(500)
        )
        self.save_pdb(self._prefix + "_minimized.pdb")
        self.simulation.step(steps)

        return self.simulation

    def restart(self, steps: int = 50000000, restart_file: str = None) -> app.Simulation:
        '''
        Run the simulation from given pdb file. Default: 50 million steps (100 ns).

        :param steps: Number of steps to run the simulation.
            Default: 50 million steps (100 ns)
        :type steps: int
        :param restart_file: Name of the checkpoint file. Default: {prefix}.chk
        :type restart_file: str
        '''

        self._load_checkpoint(restart_file)
        self._add_reporter(self._get_thermo_reporter(steps))
        self.simulation.step(steps)

        return self.simulation

    def save_checkpoint(self, filename: str = None) -> None:
        '''
        Save the checkpoint of the simulation.

        :param filename: Name of the checkpoint file. Default: `{prefix}_out.chk`
        :type filename: str
        '''
        if filename is None:
            filename = f"{self._prefix}_out.chk"
        self.simulation.saveCheckpoint(filename)

    def save_pdb(self, filename: str = None) -> None:
        '''
        Save the final state PDB of the simulation.

        :param filename: Name of the pdb file. Default: `{prefix}_out.pdb`
        :type filename: str
        '''
        if filename is None:
            filename = f"{self._prefix}_out.pdb"
        with open(filename, 'w') as ofs:
            # Note that this pos has no underscore, so it will call the property
            # and get the latest positions from the simulation context
            app.PDBFile.writeFile(self._top, self.pos, ofs, keepIds=True)

    def _load_checkpoint(self, restart_file: str) -> None:
        '''
        Load the checkpoint of the simulation.

        :param filename: Name of the checkpoint file.
        :type filename: str
        '''

        if restart_file is None:
            restart_file = self._prefix + ".chk"
            print(f"No restart file provided. Attempting from default {restart_file}.")

        if not os.path.exists(restart_file):
            raise FileNotFoundError("Checkpoint file does not exist")

        if restart_file.endswith(".chk"):
            self.simulation.loadCheckpoint(restart_file)
        elif restart_file.endswith(".pdb"):
            pdb = app.PDBFile(restart_file)
            self.simulation.context.setPositions(pdb.positions)

    def _initialize_simulation(self, **kwargs) -> app.Simulation:
        '''
        Create the integrator for the system using LangevinMiddleIntegrator.
        Returns the OpenMM simulation object.

        :param dt: Time step of the simulation. Default: 0.002, in ps
        :type dt: openmm.unit.
        :param cutoff: Nonbonded cutoff. Default: 10.0, in Angstroms
        :type cutoff: float
        '''

        dt = kwargs.get('dt', 0.002 * picoseconds)
        if not is_quantity(dt):
            dt = dt * picoseconds

        cutoff = kwargs.get('cutoff', 10.0 * angstroms)
        if not is_quantity(cutoff):
            cutoff = cutoff * angstroms

        system = self._forcefield.createSystem(
            self._top,
            nonbondedMethod=app.PME,
            nonbondedCutoff=cutoff,
            constraints=app.HBonds
        )
        integrator = openmm.LangevinMiddleIntegrator(self._temp, 1 / picoseconds, dt)
        simulation = app.Simulation(self._top, system, integrator, self._platform)

        return simulation

    def _choose_platform(self) -> openmm.Platform:
        '''
        Choose the platform for the simulation.
        1. CUDA 2. OpenCL 3. CPU

        :return: OpenMM platform
        :rtype: OpenMM.Platform
        :raises: RuntimeError if no suitable platform is found.
        '''

        # enumerate all platforms with openmm
        n_platforms = openmm.Platform.getNumPlatforms()
        platforms = [openmm.Platform.getPlatform(i) for i in range(n_platforms)]
        platform_names = [platform.getName() for platform in platforms]

        # We will use platforms in the following order
        selection = ["CUDA", "OpenCL", "CPU"]
        for plt in selection:
            if plt in platform_names:
                print(f"Using {plt} platform.")
                return platforms[platform_names.index(plt)]

        # if the code reaches here something is wrong
        raise RuntimeError("No suitable platform found. Attempted: CUDA, OpenCL, CPU")

    def _get_thermo_reporter(self, steps: int) -> None | app.StateDataReporter:
        '''
        Initialize the state reporters for the simulation.
        '''

        if self._progress_every is None:
            return None
        rep = app.StateDataReporter(stdout,
                                    self._progress_every, step=True,
                                    potentialEnergy=True, temperature=True, volume=True,
                                    progress=True, remainingTime=True,
                                    totalSteps=steps, elapsedTime=True,
                                    speed=True, separator="\t", append=self._append
                                    )
        return rep

    def _get_cv_reporter(self, **kwargs) -> CVReporter | None:
        '''
        Get the CV reporter for the simulation.

        :return: CVReporter object or None
        :rtype: CVReporter | None
        '''

        if "cv_reporter" in kwargs:
            return kwargs["cv_reporter"]

        list_of_index = kwargs.get('list_of_index', None)

        if list_of_index is not None:
            cv_file = kwargs.get('cv_file', self._prefix + "_colvar.dat")
            cv_freq = kwargs.get('cv_freq', 50)
            append = kwargs.get('append', False)
            return CVReporter(cv_file, cv_freq, list_of_index, append)

        print("No atom indices provided. Will not output CV timeseries.")
        return None

    def _get_xtc_reporter(self, **kwargs) -> XTCReporter | None:
        '''
        Get the CV reporter for the simulation.

        :return: CVReporter object or None
        :rtype: CVReporter | None
        '''

        if "xtc_reporter" in kwargs:
            return kwargs["xtc_reporter"]
        else:
            xtc_file = kwargs.get('xtc_file', self._prefix + ".xtc")
            xtc_freq = kwargs.get('xtc_freq', 1000)

            # We will allow the user to suppress trajectory writing by setting
            # either traj_filename or traj_freq to None
            if (xtc_file is None) or (xtc_freq is None):
                return None

            return XTCReporter(xtc_file, xtc_freq, append=self._append)

    def _get_pressure(self, **kwargs) -> Quantity | None:
        '''
        Get the pressure for the simulation.

        :return: Pressure of the system
        :rtype: openmm.unit.Quantity[unit=bar]
        '''

        pressure = kwargs.get('pressure', 1.0 * bar)
        if pressure is None:
            return None
        if not is_quantity(pressure):
            return pressure * bar
        return pressure

    def _get_temperature(self, **kwargs) -> Quantity:
        '''
        Get the temperature for the simulation.

        :return: Temperature of the system
        :rtype: openmm.unit.Quantity[unit=kelvin]
        '''

        temp = kwargs.get('temp', 310.0 * kelvin)
        if temp is None:
            raise ValueError("Temperature cannot be set to None.")
        if not is_quantity(temp):
            return temp * kelvin
        return temp

    def _add_reporter(self, reporter) -> None:
        '''
        Add a reporter to the simulation. Allows a NoneType reporter which is ignored.

        :param reporter: Reporter object
        :type reporter: openmm.app.StateDataReporter | CVReporter | XTCReporter
        '''

        if reporter is None:
            pass
        else:
            self.simulation.reporters.append(reporter)
