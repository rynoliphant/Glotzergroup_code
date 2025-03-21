import numpy as np
import coxeter


vertices = np.array([
    [0,1],[1,0],[1,1],[-1,-1]
])

shape = coxeter.shapes.ConvexPolygon(vertices, normal=None, planar_tolerance=1e-05)

print(shape.vertices[:,:2])

shp_vert = shape.vertices[:,:2]
n_verts = len(shp_vert)

print(shp_vert - np.append(shp_vert[1:], shp_vert[0].reshape(1,2), axis=0))

shp_edges = shp_vert - np.append(shp_vert[1:], shp_vert[0].reshape(1,2), axis=0) #edges point clockwise

shp_edge_vert = np.append(np.arange(1,n_verts +1).reshape(n_verts,1), np.arange(0,n_verts).reshape(n_verts,1), axis=1) #column [0]: o-->  | column [1]: o<--   | shp.edges[1] - shp.edges[0] == shp.edge_vectors
shp_edge_vert[-1:0] = 0

print(shp_edge_vert.shape)


#----- Getting the constraint matrix and bounds needed to define the space partitioned zones -----
vert_constraint = np.append( shp_edges.reshape(n_verts, 1,2), -1*np.append(shp_edges[1:], shp_edges[0].reshape(1,2), axis=0).reshape(n_verts,1,2), axis=1)

print(vert_constraint.shape)
print(vert_constraint)