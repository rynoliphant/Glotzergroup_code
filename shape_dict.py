import numpy as np
import sys

#input
inputs = sys.argv

if len(inputs) == 1:
    shape_type = input('Shape type?: ')
    shape_a = 0
    custom_a = input('Do you wish to set a custom length value (y/n)?: ')
    if custom_a == 'y':
        shape_a = input('Length of a?: ')
elif len(inputs) == 2:
    shape_type = inputs[1]
    shape_a = 0
elif len(inputs) == 3:
    shape_type = inputs[1]
    shape_a = inputs[2]
else:
    print(f'You provided extra arguments! Only {inputs[1]} and {inputs[2]} will be used!')
    shape_type = inputs[1]
    shape_a = inputs[2]

#shape dictionary
#origin is set as the center of the shape
#the height of the shape is 1
shape_vertices = {'cube': [
    (0.5, 0.5, 0.5),
    (-0.5, 0.5, 0.5),
    (-0.5, -0.5, 0.5),
    (-0.5, 0.5, -0.5),
    (0.5, -0.5, 0.5),
    (0.5, -0.5, -0.5),
    (0.5, 0.5, -0.5),
    (-0.5, -0.5, -0.5),
], 'octahedron': [
    (-0.5, 0, 0),
    (0.5, 0, 0),
    (0, -0.5, 0),
    (0, 0.5, 0),
    (0, 0, -0.5),
    (0, 0, 0.5),
], 'tetrahedron': [
    (0, 0, 0.5),
    (0, 0.7071, -0.5),
    (0.5*np.sqrt(3/2), -0.3535, -0.5),
    (-0.5*np.sqrt(3/2), -0.3535, -0.5),
], 'dodecahedron': [
    (0.5, 0.5, 0.5),
    (-0.5, 0.5, 0.5),
    (-0.5, -0.5, 0.5),
    (-0.5, 0.5, -0.5),
    (0.5, -0.5, 0.5),
    (0.5, -0.5, -0.5),
    (0.5, 0.5, -0.5),
    (-0.5, -0.5, -0.5),
    (0, 0.5/((1+np.sqrt(5))*0.5), 0.5*((1+np.sqrt(5))*0.5)),
    (0, -0.5/((1+np.sqrt(5))*0.5), 0.5*((1+np.sqrt(5))*0.5)),
    (0, 0.5/((1+np.sqrt(5))*0.5), -0.5*((1+np.sqrt(5))*0.5)),
    (0, -0.5/((1+np.sqrt(5))*0.5), -0.5*((1+np.sqrt(5))*0.5)),
    (0.5/((1+np.sqrt(5))*0.5), 0.5*((1+np.sqrt(5))*0.5), 0),
    (-0.5/((1+np.sqrt(5))*0.5), 0.5*((1+np.sqrt(5))*0.5), 0),
    (0.5/((1+np.sqrt(5))*0.5), -0.5*((1+np.sqrt(5))*0.5), 0),
    (-0.5/((1+np.sqrt(5))*0.5), -0.5*((1+np.sqrt(5))*0.5), 0),
    (0.5*((1+np.sqrt(5))*0.5), 0, 0.5/((1+np.sqrt(5))*0.5)),
    (-0.5*((1+np.sqrt(5))*0.5), 0, 0.5/((1+np.sqrt(5))*0.5)),
    (0.5*((1+np.sqrt(5))*0.5), 0, -0.5/((1+np.sqrt(5))*0.5)),
    (-0.5*((1+np.sqrt(5))*0.5), 0, -0.5/((1+np.sqrt(5))*0.5)),
]}

#a values so that the height of the shape is 1 unit
a_values = {'cube': 1,
            'octahedron': np.sqrt(2) * 0.5,
            'tetrahedron': np.sqrt(3/2)
}

if float(shape_a) != 0:
    try:
        a_values[shape_type]
    except:
        print('That shape is not among the list of shapes available! :(')
    else:
        size_multiplier = float(shape_a) / a_values[shape_type]
        a_values[shape_type] = float(shape_a)
else:
    size_multiplier = 1

#volume and surface area dictionaries
shape_volumes = {'cube': a_values['cube']**3, 
                 'octahedron': 1/3 * np.sqrt(2) * a_values['octahedron']**3,
                 'tetrahedron': a_values['tetrahedron']**3 / (6*np.sqrt(2))
}

shape_surface_area = {'cube': 6*a_values['cube']**2,
                      'octahedron': 2*np.sqrt(3) * a_values['octahedron']**2,
                      'tetrahedron': np.sqrt(3) * a_values['tetrahedron']**2
}


#output
try:
    shape_vertices[shape_type]
except:
    print()
else:
    print('Vertices: \n', np.array(shape_vertices[shape_type])*size_multiplier)
    print('a Value: ', round(a_values[shape_type], 3))
    print('Volume: ', round(shape_volumes[shape_type], 3))
    print('Surface Area: ', round(shape_surface_area[shape_type], 3))
    print('Isopermetric Quotient: ', round(((36 * np.pi * shape_volumes[shape_type]**2) / (shape_surface_area[shape_type]**3)), 3))
