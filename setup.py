from setuptools import setup,find_packages

def get_version():
    with open("cxdi/__init__.py", "r") as fid:
        lines = fid.readlines()
    version = None
    for line in lines:
        if "version" in line:
            version = line.rstrip().split("=")[-1].lstrip()
    if version is None:
        raise RuntimeError("Could not find version from __init__.py")
    version = version.strip("'").strip('"')
    return version


setup(name='cxdi',
      version=get_version(),
      description='utilities for Coherent Xray',
      url='https://github.com/marcocamma/cxdi',
      author='marco cammarata',
      author_email='marco.cammarata@esrf.eu',
      license='MIT',
      packages=find_packages("."),
      install_requires=[
          'numpy',
          'silx',
      ],
      include_package_data=True,
      zip_safe=False)
