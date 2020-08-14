import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="multi_armed_thompson",
    version="0.2.1",
    license='MIT',
    author="Tullio Bagnoli",
    author_email="tullio.bagnoli@protonmail.com",
    description="Thompson sampling for multi-armed bandit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tbagnoli/multi_armed_thompson/",
    download_url='https://github.com/tbagnoli/multi_armed_thompson/archive/v_01.tar.gz',
    keywords=['thompson', 'sampling', 'unsupervised'],
    packages=setuptools.find_packages(where='src'),
    package_dir={
        '': 'src'
    },
    install_requires=['numpy', 'spacy'],
    include_package_data=True,  # forces inclusion of files in MANIFEST.in
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
