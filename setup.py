from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='mlnowcasting',
    version='0.0.0',
    description='Machine Learning models for weather nowcasting',
    #long_description=readme,
    author='Joaquin Cuomo',
    #url='https://github.com/kennethreitz/samplemod',
     classifiers=[  # Optional
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Researchers',
        'Topic :: Weather Nowcasting :: Video Prediction',
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by 'pip install'. See instead 'python_requires' below.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
    ],
    package_dir={'': 'mlnowcasting'},  
    packages=find_packages(where='mlnowcasting'),
    python_requires='>=3.5, <4',
    install_requires=['matplotlib', 'numpy', 'pandas', 'scikit-image', 'scipy'],  # Optional
    license=license,
)
