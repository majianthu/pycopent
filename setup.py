import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="copent",
    version="0.5.4",
    author="MA Jian",
    author_email="majian03@gmail.com",
    description="Estimating Copula Entropy and Transfer Entropy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license = 'GPL License',
    url="https://github.com/majianthu/pycopent",
    packages=setuptools.find_packages(),
    install_requires = ['numpy','scipy'],
    python_requires='>=2.7',
)
