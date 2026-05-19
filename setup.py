from setuptools import setup, find_packages

setup(
    name='metetl',
    version='0.1',
    py_modules=['artwork', 'decade_stats', 'logging_config', 'cli'],
    entry_points={
        'console_scripts': [
            'metetl = cli:main',
        ],
    },
)