import numpy as np
import numpy.linalg as LA

x_coord = np.array([[0,1,0], [2,2,3]])

s_position = np.array([0,0,0])


x = np.transpose(x_coord)

A = np.array([ [[1,0,0],[0,1,0],[0,0,1],[1,1,0]],  [[-1,0,0],[1,-1,0],[1,1,1],[0,1,-1]],  [[-1,-1,1],[0,-1,-1],[0,1,0],[1,0,1]],  [[1,-1,-1],[-1,0,-1],[-1,1,0],[0,0,1]]])

#print(A @ x)

a = np.reshape(np.array([[2,3,2,3],[0,0,8,2],[0,-2,2,5],[-1,0,-1,-1]]), (4,4,1))

#print(a)

vert_bool = np.all((A@x)<= a, axis=1)

print(np.transpose(vert_bool))


a_verts = np.array([ [0,0,2], [1,2,0], [0,1,0.5], [0,1,0]])

max_value = 3*np.max(LA.norm(x_coord - s_position, axis=1))

vert_any_bool = np.any(vert_bool, axis=1)

print(vert_any_bool)
print(a_verts[vert_any_bool])
print(np.repeat(np.reshape(x_coord, (len(x_coord),1, 3)), len(a_verts[vert_any_bool]), axis=1))

#v--- Edge Distance ---v



#v--- Vertex Distance ---v
vert_dist = LA.norm(np.repeat(np.reshape(x_coord, (len(x_coord),1, 3)), len(a_verts[vert_any_bool]), axis=1) - (a_verts[vert_any_bool] + s_position), axis=2)

vert_dist = vert_dist + max_value*(np.transpose(vert_bool[vert_any_bool]) == False).astype(int)

vert_argmin = np.argmin(vert_dist, axis=1)

vert_dist = np.min(vert_dist, axis=1)

# vert_displacement = (np.repeat(np.reshape(x_coord, (len(x_coord),1, 3)), len(a_verts[vert_any_bool]), axis=1) - a_verts[vert_any_bool])[vert_argmin]
# print('displacement:',vert_displacement)

print(vert_dist)

print(np.concatenate((vert_dist.reshape(2,1), vert_dist.reshape(2,1)), axis=1))

big_arr = np.array([[[1,1,1], [0,0,1]],  [[1,1,1], [0,0,1]],  [[1,1,1], [0,0,1]]]) #(3,2,3)

small_arr = np.reshape(np.array([[3,0,0], [0,3,0], [0,0,3]]), (3,1,3)) #(3,1,3)

other_arr = np.array([[0,1,0], [-1,0,0]]) #(2,3)

print(big_arr - small_arr)

print(np.reshape(np.sum((big_arr -small_arr) * other_arr, axis=2), (3,2,1)) * np.repeat(np.reshape(other_arr, (1,2,3)), 3, axis=0)   )