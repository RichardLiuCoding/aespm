from setuptools import setup, find_packages

setup(
  name = 'aespm',
  version = '1.0.0',
  packages = find_packages(),
  license='MIT',
  description = 'Asylum Research SPM data analysis packages',
  author = 'Richard (Yu) Liu',
  author_email = 'yliu206@utk.edu',
  url = 'https://github.com/RichardLiuCoding/aespm',
  download_url = 'https://github.com/RichardLiuCoding/aespm.git',
  keywords = ['SPM', 'AR', 'Python', 'Instrument control', 'Autonomoous', 'Machine learning','Data Analysis'],
  install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'igor2',
          'paramiko',
          'Pyro5',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3'
  ],
)
