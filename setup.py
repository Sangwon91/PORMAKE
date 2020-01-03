from setuptools import setup

setup(
    name="pormake",
    version="0.0.1",
    description="Construction of nanoporous materials"
                " from topology and building blocks.",
    install_requires=[
        "numpy",
        "scipy",
        "pymatgen",
        "ase==3.18.0",
        "tensorflow>=2.0.0",
    ],
    author="Sangwon Lee",
    author_email="integratedmailsystem@gmail.com",
    packages=[
        "pormake",
        "pormake.third_party",
        "pormake.third_party.rmsd",
        "pormake.third_party.rmsd.rmsd",
    ],
    zip_safe=False
)
