from setuptools import find_packages, setup

setup(name='src', 
        packages=find_packages(),
        version='0.1.0',
      
        extras_require={
                'phase-label-network': ['tensorflow==2.13', 'keras', 'seaborn']
        }
      )
