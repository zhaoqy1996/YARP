from ase import Atoms

# Setup a chain of H,O,C
# H-O Dist = 2
# O-C Dist = 3
# C-H Dist = 5 with mic=False
# C-H Dist = 4 with mic=True
a = Atoms('HOC', positions=[(1, 1, 1), (3, 1, 1), (6, 1, 1)])
a.set_cell((9, 2, 2))
a.set_pbc((True, False, False))


# Calculate indiviually with mic=True
assert a.get_distance(0, 1, mic=True) == 2
assert a.get_distance(1, 2, mic=True) == 3
assert a.get_distance(0, 2, mic=True) == 4

# Calculate indiviually with mic=False
assert a.get_distance(0, 1, mic=False) == 2
assert a.get_distance(1, 2, mic=False) == 3
assert a.get_distance(0, 2, mic=False) == 5

# Calculate in groups with mic=True
assert (a.get_distances(0, [1, 2], mic=True) == [2, 4]).all()

# Calculate in groups with mic=False
assert (a.get_distances(0, [1, 2], mic=False) == [2, 5]).all()

# Calculate all with mic=True
assert (a.get_all_distances(mic=True) == [[0, 2, 4],
                                          [2, 0, 3],
                                          [4, 3, 0]]).all()

# Calculate all with mic=False
assert (a.get_all_distances(mic=False) == [[0, 2, 5],
                                           [2, 0, 3],
                                           [5, 3, 0]]).all()

# Scale Distance
old = a.get_distance(0, 1)
a.set_distance(0, 1, 0.9, add=True, factor=True)
new = a.get_distance(0, 1)
diff = new - 0.9 * old
assert abs(diff) < 10e-6

# Change Distance
old = a.get_distance(0, 1)
a.set_distance(0, 1, 0.9, add=True)
new = a.get_distance(0, 1)
diff = new - old - 0.9
assert abs(diff) < 10e-6
