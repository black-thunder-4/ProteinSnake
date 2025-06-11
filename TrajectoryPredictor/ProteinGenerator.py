from openmm.app import Topology, PDBReporter, StateDataReporter, Simulation, Element
from openmm import System, CustomBondForce, CustomAngleForce, CustomTorsionForce, CustomNonbondedForce, LangevinIntegrator, Vec3
from openmm.unit import nanometer, kilojoule_per_mole, picosecond, kelvin, radian, dalton
import numpy as np
import random
import sys
import os

def generate_multiple_proteins(sequence_lengths, residue_types = ["ALA", "GLY"], out_path='trajectories'):
    for length in sequence_lengths:
        generate_protein(length, residue_types=residue_types, out_path=out_path)

def generate_protein(sequence_length, residue_types = ["ALA", "GLY"], out_path='trajectories'):

    # Generate a sequence of 100 residues chosen at random.
    sequence = [random.choice(residue_types) for _ in range(sequence_length)]

    topology = Topology()
    chain = topology.addChain()
    positions = []
    # Also store a numeric value for each residue type: 0.0 for ALA, 1.0 for GLY.
    residue_numeric = []

    for i, res_name in enumerate(sequence):
        residue = topology.addResidue(res_name, chain)
        # Add a single atom (using a carbon as a placeholder for the Cα).
        topology.addAtom("CA", Element.getByAtomicNumber(6), residue)
        
        # Assign a numeric type for later use in the force.
        res_type_num = 0.0 if res_name == "ALA" else 1.0
        residue_numeric.append(res_type_num)
        
        # Place atoms in an extended conformation along the x-axis with slight random noise.
        noise = Vec3(random.uniform(-0.01, 0.01),
                     random.uniform(-0.01, 0.01),
                     random.uniform(-0.01, 0.01)) * nanometer
        positions.append(Vec3(i * 0.38, 0, 0) * nanometer + noise)

    system = System()
    for i in range(sequence_length):
        system.addParticle(12.0 * dalton)

    # Bond parameters (target distance ~0.38 nm).
    r0 = 0.38 * nanometer  
    k_bond = 1000 * kilojoule_per_mole / nanometer**2
    
    # Angle parameters (target angle ~105° or 1.83 rad).
    angle0 = 1.83 * radian     
    k_angle = 100 * kilojoule_per_mole / radian**2
    
    # Torsion parameters (target dihedral ~100° or 1.745 rad).
    dihedral0 = 1.745 * radian  
    k_dihedral = 1 * kilojoule_per_mole
    
    # Lennard-Jones parameters for the generic nonbonded force.
    epsilon_nb = 5 * kilojoule_per_mole  
    sigma_nb = 0.3 * nanometer
    
    # Parameters for the residue-specific contact potential.
    r_contact = 0.7 * nanometer
    delta = 0.1 * nanometer
    rmin = 0.5 * nanometer
    rmax = 1.5 * nanometer
    epsilon_attract = 5 * kilojoule_per_mole  # attractive strength for same residues.
    epsilon_repulse = 5 * kilojoule_per_mole  # repulsive strength for different residues.

    # Bond Force: harmonic bonds between adjacent CA atoms.
    bond_force = CustomBondForce("0.5 * k_bond * (r - r0)^2")
    bond_force.addPerBondParameter("k_bond")
    bond_force.addPerBondParameter("r0")
    for i in range(sequence_length - 1):
        bond_force.addBond(i, i+1, [k_bond, r0])
    system.addForce(bond_force)
    
    # Angle Force: harmonic angles for each set of three consecutive CA atoms.
    angle_force = CustomAngleForce("0.5 * k_angle * (theta - angle0)^2")
    angle_force.addPerAngleParameter("k_angle")
    angle_force.addPerAngleParameter("angle0")
    for i in range(sequence_length - 2):
        angle_force.addAngle(i, i+1, i+2, [k_angle, angle0])
    system.addForce(angle_force)
    
    # Torsion Force: cosine potential for dihedrals defined by four consecutive CA atoms.
    torsion_force = CustomTorsionForce("0.5 * k * (1 - cos(theta - theta0))")
    torsion_force.addPerTorsionParameter("k")
    torsion_force.addPerTorsionParameter("theta0")
    for i in range(sequence_length - 3):
        torsion_force.addTorsion(i, i+1, i+2, i+3, [k_dihedral, dihedral0])
    system.addForce(torsion_force)
    
    # Residue-Specific Contact Force
    # This custom nonbonded force differentiates interactions based on residue type.
    # For a pair of particles:
    #   - If they are of the same type (abs(res_type - res_type2) == 0), the force is attractive (-epsilon_attract).
    #   - If they are different (abs(res_type - res_type2) == 1), the force is repulsive (+epsilon_repulse).
    #
    # The force is only active between distances rmin and rmax and is modulated by a Gaussian
    # centered at r_contact with width delta.
    
    contact_expression = (
        "step(r - rmin)*step(rmax - r)*(" 
        "   ((1 - abs(res_type1 - res_type2)) * (-epsilon_attract) + abs(res_type1 - res_type2) * (epsilon_repulse))"
        "   * exp(-((r - r_contact)^2)/(delta^2))"
        ")"
    )
    hetero_force = CustomNonbondedForce(contact_expression)
    hetero_force.addGlobalParameter("epsilon_attract", epsilon_attract)
    hetero_force.addGlobalParameter("epsilon_repulse", epsilon_repulse)
    hetero_force.addGlobalParameter("r_contact", r_contact)
    hetero_force.addGlobalParameter("delta", delta)
    hetero_force.addGlobalParameter("rmin", rmin)
    hetero_force.addGlobalParameter("rmax", rmax)
    # Add a per-particle parameter called 'res_type'.
    hetero_force.addPerParticleParameter("res_type")
    for i in range(sequence_length):
        hetero_force.addParticle([residue_numeric[i]])
    # Exclude interactions between directly bonded neighbors.
    for i in range(sequence_length - 1):
        hetero_force.addExclusion(i, i+1)
    hetero_force.setCutoffDistance(1.0 * nanometer)
    hetero_force.setNonbondedMethod(CustomNonbondedForce.CutoffNonPeriodic)
    system.addForce(hetero_force)

    # Lennard-Jones Nonbonded Force (Generic)
    nonbonded_force = CustomNonbondedForce("4*epsilon*((sigma/r)^12 - (sigma/r)^6)")
    nonbonded_force.addGlobalParameter("epsilon", epsilon_nb)
    nonbonded_force.addGlobalParameter("sigma", sigma_nb)
    nonbonded_force.setCutoffDistance(1.0 * nanometer)
    nonbonded_force.setNonbondedMethod(CustomNonbondedForce.CutoffNonPeriodic)
    for i in range(sequence_length):
        nonbonded_force.addParticle([])
    for i in range(sequence_length - 1):
        nonbonded_force.addExclusion(i, i+1)
    system.addForce(nonbonded_force)
    
    # Set up the Simulation with Simulated Annealing.
    initial_temp = 300 * kelvin
    final_temp = 10 * kelvin
    n_intervals = 20
    steps_per_interval = 2500  # Total simulation steps: 20 * 2500 = 50,000
    
    integrator = LangevinIntegrator(initial_temp, 1/picosecond, 0.002*picosecond)
    simulation = Simulation(topology, system, integrator)
    # Directly set the positions (no need to write/read a file).
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy()
    
    # Reporters: save a PDB snapshot every 50 steps and log simulation data.
    file_name = ''.join([res[0] for res in sequence])
    file = os.path.join(out_path, f"{file_name}.pdb")
    simulation.reporters.append(PDBReporter(file, 50))
    
    temps = np.linspace(initial_temp.value_in_unit(kelvin), final_temp.value_in_unit(kelvin), n_intervals) * kelvin
    for T in temps:
        integrator.setTemperature(T)
        simulation.step(steps_per_interval)
    
    print(f"Simulation complete. Check {file} for the trajectory.")