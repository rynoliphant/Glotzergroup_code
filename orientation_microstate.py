import numpy as np
import plotly.express as px
import sys
inputs = sys.argv

#New attempt

def new_points (pr, u, n):
    '''
    pr (array): parent point
    u (array): axis of rotation
    n (int): 
    '''
    theta = (2*np.pi) / n

    rot_points = []

    for ni in range(1,n):
        theta_i = ni * theta

        R = np.array([
            [u[0]*u[0]*(1-np.cos(theta_i)) + np.cos(theta_i),      u[0]*u[1]*(1-np.cos(theta_i)) - u[2]*np.sin(theta_i), u[0]*u[2]*(1-np.cos(theta_i)) + u[1]*np.sin(theta_i)],
            [u[0]*u[1]*(1-np.cos(theta_i)) + u[2]*np.sin(theta_i), u[1]*u[1]*(1-np.cos(theta_i)) + np.cos(theta_i),      u[1]*u[2]*(1-np.cos(theta_i)) - u[0]*np.sin(theta_i)],
            [u[0]*u[2]*(1-np.cos(theta_i)) - u[1]*np.sin(theta_i), u[1]*u[2]*(1-np.cos(theta_i)) + u[0]*np.sin(theta_i), u[2]*u[2]*(1-np.cos(theta_i)) + np.cos(theta_i)]
        ])

        rot_points.append(R @ pr)

    return np.asarray(rot_points)

def set_up (n):
    p0 = np.array([0,0,1])
    theta = (2*np.pi) / n
    u = np.array([0,1,0])

    R = np.array([
            [u[0]*u[0]*(1-np.cos(theta)) + np.cos(theta),      u[0]*u[1]*(1-np.cos(theta)) - u[2]*np.sin(theta), u[0]*u[2]*(1-np.cos(theta)) + u[1]*np.sin(theta)],
            [u[0]*u[1]*(1-np.cos(theta)) + u[2]*np.sin(theta), u[1]*u[1]*(1-np.cos(theta)) + np.cos(theta),      u[1]*u[2]*(1-np.cos(theta)) - u[0]*np.sin(theta)],
            [u[0]*u[2]*(1-np.cos(theta)) - u[1]*np.sin(theta), u[1]*u[2]*(1-np.cos(theta)) + u[0]*np.sin(theta), u[2]*u[2]*(1-np.cos(theta)) + np.cos(theta)]
        ])
    
    pr = np.round(R @ p0, 10)

    points = new_points(pr, p0, n)

    initial = np.append(p0.reshape(1,3), pr.reshape(1,3), axis=0)
    initial_points = np.append(initial, points, axis=0)

    return initial_points


n = int(inputs[1])
points = set_up(n)
parent_int = np.zeros(len(points)).astype(int)
not_complete = np.ones(len(points)).astype(bool)
not_complete[0] = False
old_len = 1

print(points)

init_points = np.append(points, np.array([[0,0,0]]), axis=0)
fig = px.scatter_3d(x=init_points[:,0], y=init_points[:,1], z=init_points[:,2])
fig.show()


while len(points) != old_len:
    old_len = len(points)

    if len(points) >= 50:
        break

    if np.any(not_complete):
        incomplete_ind = np.arange(0,len(points),1)[not_complete]
        # print(parent_int)
        # print(points)
        # print(points.astype(np.float16))

        for p_i, point in enumerate(points[not_complete]):
            all_p_i = incomplete_ind[p_i]
            parent = points[parent_int[all_p_i]]

            # print('parent',parent)
            # print('rot axis', point)

            # print('Points match?', point == points[all_p_i])

            if p_i == 0:
                plus_points = new_points(parent, point, n)
                new_parent_int = (np.ones(len(plus_points)) * all_p_i).astype(int)
                not_complete[all_p_i] = False
            else:
                add_points = new_points(parent, point, n)
                plus_points = np.append(plus_points, add_points, axis=0)
                new_parent_int = (np.append(new_parent_int, np.ones(len(add_points)) * all_p_i)).astype(int)
                not_complete[all_p_i] = False
            
        # print(plus_points)

        not_complete = np.append(not_complete, np.ones(len(plus_points))).astype(bool)
        parent_int = np.append(parent_int, new_parent_int).astype(int)
        points = np.append(points, plus_points, axis=0)

        #unique
        ignore, unique_inds = np.unique(points.astype(np.float32), return_index=True, axis=0)
        points = points[unique_inds]
        not_complete = not_complete[unique_inds]
        parent_int = parent_int[unique_inds]

        print(len(points))


    else:
        break


print(len(points))
print(points)

points = np.append(points, np.array([[0,0,0]]), axis=0)

fig = px.scatter_3d(x=points[:,0], y=points[:,1], z=points[:,2])
fig.show()



#spherical coordinates
'''
p0 = np.array([1.0, 0.0, 0.0]) #initial point
pi_factor = np.array([1, np.pi, np.pi])
n = int(inputs[1])

def create_new_points (n):
    phi_rotations = (np.arange(0, 2, 2/n)).reshape(n, 1)
    zeros = np.zeros((n, 1))
    ones = np.ones((n, 1))
    zero_points = np.concatenate((ones, zeros, phi_rotations), axis=1)
    theta_points = np.concatenate((ones, (2/n)*ones, phi_rotations), axis=1)
    new_points = np.concatenate((zero_points, theta_points), axis=0)

    return new_points

def rotate_all (point, unique_points, p0):
    diff = p0 - point
    print('diff',diff)
    unique_points += diff
    if np.any(unique_points >= np.array([1.1,1,2])):
        bool = (unique_points >= np.array([1,1,2]))*np.array([0,1,2])
        unique_points = unique_points - bool

    if np.any(unique_points < np.array([1,0,0])):
        neg_bool = unique_points < np.array([1,0,0])
        unique_points = unique_points - 2*(neg_bool * unique_points * np.array([0,1,0])) + 2 * neg_bool * np.array([0,0,1])

    return unique_points



unique_points = np.array([p0])
old_len = 0
each_new_points = create_new_points (n)

while len(unique_points) != old_len:
    old_len = len(unique_points)
    for point in unique_points:
        unique_points = rotate_all(point, unique_points, p0)
        print('post rotate:', unique_points)
        unique_points = np.append(unique_points,each_new_points, axis=0)

        # unique_points = np.round(unique_points, 5)

        unique_points = np.unique(unique_points, axis=0)

        # print('unique_points:',unique_points)

        if len(unique_points) > 30:
            print('Uh oh, unique_points is too long!!!', len(unique_points))
            break
    if len(unique_points) > 30:
        break

print(unique_points)
print(len(unique_points))
print(each_new_points)


#delete later/ignore
coords = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
shape_pos = np.array([3,3,3])
ijk_list = np.array([
        [-1,-1,-1],[-1,-1,0],[-1,-1,1],
        [-1,0,-1],[-1,0,0],[-1,0,1],
        [-1,1,-1],[-1,1,0],[-1,1,1],
    ])

image_pos = ijk_list + shape_pos
big_image_pos = np.tile(image_pos.reshape(9,1,3), (1,4,1))
print(big_image_pos.shape)
big_coords = np.tile(coords.reshape(1,4,3), (9,1,1))
print(big_coords.shape)
dist_arr = np.sqrt(np.sum((big_image_pos-big_coords)**2, axis=2))
print(dist_arr.shape)
min_dist = np.min(dist_arr, axis=0)
print(min_dist.shape)


# madeuplist = [1,3,4]

# for i in madeuplist:
#     if i%2 == 1:
#         madeuplist = (np.asarray(madeuplist) + 2).tolist()
#         madeuplist.append(i+1)

#     print(i)

# print(madeuplist)
'''