from setuptools import setup

setup(
    name='correlation_clustering',
    version='1.0.0',    
    description='A python implementation of Correlation Clustering',
    url='https://github.com/Garrafao/correlation_clustering',
    author='Dominik Schlechtweg,
    author_email='dominik.schlechtweg@ims.uni-stuttgart.de',
    license='BSD 3-clause',
    packages=['pyexample'],
    install_requires=['mpi4py&gt;=2.0',
                      'numpy',                     
                      ],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
    ],
)
