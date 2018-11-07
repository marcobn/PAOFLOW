from distutils.core import setup
import os

defs = os.path.join('src','defs')
pys = os.listdir(defs)
rml = []

for i in range(len(pys)):
  v = pys[i]
  if v[0] == '_' or v[-3:] != '.py':
    rml.append(v)
  else:
    pys[i] = v[:-3]

for v in rml:
  pys.remove(v)

pkgs = ['PAOFLOW'] + pys
pkgd = { 'PAOFLOW' : 'src' }
for p in pys:
  pkgd[p] = defs
#print(pkgs)
#print(pkgd)

setup(name='PAOFOO',
      version='1.0',
      summary='Electronic Structure Post-processing Tools',
      author='Marco Buongiorno Nardelli',
      author_email='mbn@unt.edu',
      platforms='Linux',
      url='ERMES',
      packages=pkgs,
      package_dir=pkgd)
