import os.path
import setuptools

repository_dir = os.path.dirname(__file__)

with open(os.path.join(repository_dir, "requirements.txt")) as fh:
    requirements = [line for line in fh.readlines()]

setuptools.setup(
    name="ancer-python",
    version="1.0.0",
    author="Motasem Alfarra",
    author_email="motasem.alfarra@kaust.edu.sa",
    url="https://github.com/MotasemAlfarra/ANCER",
    license="MIT",
    python_requires=">=3.7",
    description="Anisotropic sample-wise randomized smoothing package",
    long_description="See [the project repository](https://github.com/MotasemAlfarra/ANCER) for more information.",
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7"
    ],
    install_requires=requirements,
    include_package_data=True,
)
