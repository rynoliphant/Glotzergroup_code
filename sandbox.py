import numpy as np

n_verts = 4
shp_verts = np.array([
    [0.5,0.5],
    [-0.5, 0.5],
    [-0.5,-0.5],
    [0.5,-0.5],
])

shp_x = np.array([1,2])

shp_edges = shp_verts - np.append(shp_verts[1:], shp_verts[0].reshape(1,2), axis=0) #edges point clockwise
# print(shp_edges)

shp_edge_vert = np.append(np.arange(1,n_verts +1).reshape(n_verts,1), np.arange(0,n_verts).reshape(n_verts,1), axis=1) #column [0]: o-->  | column [1]: o<--   | shp.edges[1] - shp.edges[0] == shp.edge_vectors
shp_edge_vert[-1,0] = 0
# print(shp_edge_vert)
# print(shp_edge_vert[-1,0])

vert_constraint = np.append( -1*shp_edges.reshape(n_verts, 1,2), np.append(shp_edges[-1].reshape(1,2), shp_edges[:-1], axis=0).reshape(n_verts,1,2), axis=1)
# print(vert_constraint)

vert_bounds = np.sum(vert_constraint * (shp_verts+shp_x).reshape(n_verts,1,2), axis=2)
# print(vert_bounds)
print(vert_constraint)
print('testing vert:')
print('[2, 3]', vert_bounds >= vert_constraint.dot(np.array([2,3])))
print('[-2, 3]', vert_bounds >= vert_constraint.dot(np.array([-2,3])))
print('[-2, -3]', vert_bounds >= vert_constraint.dot(np.array([-2,-3])))
print('[2, -3]', vert_bounds >= vert_constraint.dot(np.array([2,-3])))

edges_90 = np.append(shp_edges[:,1].reshape(n_verts,1), shp_edges[:,0].reshape(n_verts,1), axis=1)
edges_90 = edges_90 * np.array([1,-1])

edge_constraint = np.append( shp_edges.reshape(n_verts,1,2) , -1*shp_edges.reshape(n_verts,1,2), axis=1 )
edge_constraint = np.append( edge_constraint, edges_90.reshape(n_verts,1,2) , axis=1)
print(edge_constraint)

# edge_bounds_1 = np.sum(edge_constraint[:,0] *(edge_verts[:,1]), axis=1)
# edge_bounds_2 = np.sum(edge_constraint[:,1] *(edge_verts[:,0]), axis=1)
# edge_bounds_3 = np.sum(edge_constraint[:,2] *(edge_verts[:,0]), axis=1)

# edge_bounds[:,0] = edge_bounds_1
# edge_bounds[:,1] = edge_bounds_2
# edge_bounds[:,2] = edge_bounds_3

edge_bounds = np.zeros((n_verts, 3))

edge_bounds[:,0] = np.sum(edge_constraint[:,0] *(shp_verts+shp_x), axis=1)
edge_bounds[:,1] = np.sum(edge_constraint[:,1] *(np.append(shp_verts[1:], shp_verts[0].reshape(1,2), axis=0)+shp_x), axis=1)
edge_bounds[:,2] = np.sum(edge_constraint[:,2] *(np.append(shp_verts[1:], shp_verts[0].reshape(1,2), axis=0)+shp_x), axis=1)

print(edge_bounds)

print(0*np.array([1,2,3,4]))

lz = 0.0

print(lz!=0)