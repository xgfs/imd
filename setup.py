from setuptools import setup, find_packages


if __name__ == "__main__":
    setup(
        name="msid",
        author="Anton Tsitsulin",
        packages=find_packages(),
        install_requires=['numpy', 'scipy']
    )