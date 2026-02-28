from setuptools import setup, find_packages

setup(
    name="kaspa-tpu",
    version="0.1.0",
    description="Hybrid CPU/TPU miner for Kaspa's kHeavyHash PoW algorithm",
    packages=find_packages(include=["kaspa_tpu*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "jax>=0.4.0",
        "jaxlib>=0.4.0",
    ],
    extras_require={
        "tpu": ["jax[tpu]"],
        "grpc": ["grpcio>=1.50.0", "protobuf>=4.21.0"],
        "dev": ["pytest>=7.0.0"],
        "all": ["grpcio>=1.50.0", "protobuf>=4.21.0", "pytest>=7.0.0"],
    },
    entry_points={
        "console_scripts": [
            "kaspa-tpu=kaspa_tpu.main:main",
        ],
    },
)
