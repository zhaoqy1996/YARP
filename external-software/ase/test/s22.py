from ase.collections import s22
print(s22)
for a in s22:
    print(a)
    assert a in s22

for name in s22.names:
    assert s22.has(name)
assert not s22.has('hello')
