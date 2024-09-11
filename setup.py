from setuptools import setup, find_packages

setup(
  name = 'aespm',
  version = '1.1.3',
  packages = find_packages(),
  license='MIT',
  description = 'SPM Automation with Python and Machine Learning.',
  long_description = 'Python interface that enables local and remote control of Scanning Probe Microscope (SPM) with codes.\nIt offers a modular way to write autonomous workflows ranging from simple routine operations, to advanced automatic scientific discovery based on machine learning.',
  author = ["Richard (Yu) Liu"],
  email = ['yu93liu@gmail.com'],
  #authors = [
  #    { name="Richard (Yu) Liu", email="yu93liu@gmail.com" },
  #    { name="Boris Slautin", email="bslautin@gmail.com" },
  #],
  url = 'https://github.com/RichardLiuCoding/aespm',
  download_url = 'https://github.com/RichardLiuCoding/aespm.git',
  keywords = ['SPM', 'Python', 'Instrument control', 'Autonomoous', 'Machine learning','Data Analysis'],
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
