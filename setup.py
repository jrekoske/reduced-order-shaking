from setuptools import setup, find_packages

setup(
    name='romshake',
    version='0.1.0',
    description='Creating reduced order models of ground shaking',
    author='John Rekoske',
    author_email='jrekoske@ucsd.edu',
    url='https://github.com/jrekoske/reduced-order-shaking',
    packages=find_packages(include=['romshake', 'romshake.*']),
    scripts=['romshake/bin/build_rom']
)
