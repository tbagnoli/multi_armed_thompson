import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="thompson_sampling",
    version="0.0.1",
    author="Tullio Bagnoli",
    author_email="tullio.bagnoli@protonmail.com",
    description="Thompson sampling for multi-armed bandit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tbagnoli/multi_armed_thompson/src/thompson_sampling/thompson_sampling/",
    packages=setuptools.find_packages(where='src'),
    package_dir={
        '': 'src'
    },
    install_requires=[],
    include_package_data=True,  # forces inclusion of files in MANIFEST.in
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
