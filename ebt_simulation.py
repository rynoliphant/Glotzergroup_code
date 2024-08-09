import ase 
import coxeter
import ebonds
import freud
import gsd
import gsd.hoomd
import numpy as np

filename = 'compress_test.gsd'
frame = -10

cube_vertices = [
    (0.5, 0.5, 0.5),
    (-0.5, 0.5, 0.5),
    (-0.5, -0.5, 0.5),
    (-0.5, 0.5, -0.5),
    (0.5, -0.5, 0.5),
    (0.5, -0.5, -0.5),
    (0.5, 0.5, -0.5),
    (-0.5, -0.5, -0.5),
]

data_hpmc = gsd.hoomd.open(filename, mode='r')

orientations = data_hpmc[frame].particles.orientation
positions = data_hpmc[frame].particles.position
types = data_hpmc[frame].particles.types
typeid_list = data_hpmc[frame].particles.typeid
box = data_hpmc[frame].configuration.box

N_atoms = len(orientations)

box_tmp = freud.box.Box(Lx=box[0], Ly=box[1], Lz=box[2], xy=box[3], xz=box[4], yz=box[5])
shift_mat = np.transpose(box_tmp.to_matrix())
positions = positions + shift_mat[0,:]*0.5 + shift_mat[1,:]*0.5 + shift_mat[2,:]*0.5

atoms = ase.Atoms('C'+str(N_atoms), positions=positions,cell=shift_mat,pbc=True)

system = ebonds.data.System.from_ase_atoms(atoms)
system.particles.types = types
system.particles.type_shapes = [dict(vertices = cube_vertices)]
system.particles.typeid = typeid_list
system.particles.orientation = orientations
system.particles.position
# system = ebonds.data.System.from_gsd(filename, frame_idx=frame)

system.validate

solver = ebonds.solvers.Solver(system, phi_inp=0.6, grid_spacing=0.2)
solver.solve(mu=3, rshift=0.3, cpu_cores=3)
energy = solver.energy
print(energy)
#----- Define System -----
#sys_n (system name?)
#a
#b
#c

#alpha
#beta
#gamma

#nrep

#basis


#----- Create Periodic Box -----
#cx
#cy
#cz

#a1
#a2
#a3

#mat
#box

#uc
#box, positions

#----- Generate System for EBT Calc -----
#box1, pos1 = uc.generate_system(nrep)

#N_particles

#np.savetxt()
