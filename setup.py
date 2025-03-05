import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read() # Display description of the package if hosting a new package
    
    
__version__ = "0.0.0"

REPO_NAME = "Kidney-Disease-Detection"
AUTHOR_USERNAME = "manokrishnan123"
SRC_REPO = "cnnClassifier"
AUTHOR_EMAIL = "manobk08@gmail.com"

setuptools.setup(
    name=REPO_NAME,
    version=__version__,
    author=AUTHOR_USERNAME,
    author_email=AUTHOR_EMAIL,
    description="A simple CNN classifier to detect Kidney disease",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USERNAME}/{SRC_REPO}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USERNAME}/{SRC_REPO}/issues",
        "Documentation": f"https://github.com/{AUTHOR_USERNAME}/{SRC_REPO}/docs",
        "Source Code": f"https://github.com/{AUTHOR_USERNAME}/{SRC_REPO}/tree/main",
    },
    package_dir={"": "src"},
    packages = setuptools.find_packages(where="src")
)    