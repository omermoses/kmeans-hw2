from setuptools import setup, Extension

setup(name='mykmeanssp',
      version='1.0',
      description='kmeans for sp class',
      ext_modules=[Extension('mykmeanssp', sources=['kmeans.c'])])
