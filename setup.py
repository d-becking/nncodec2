'''
The copyright in this software is being made available under the Clear BSD
License, included below. No patent rights, trademark rights and/or
other Intellectual Property Rights other than the copyrights concerning
the Software are granted under this license.

The Clear BSD License

Copyright (c) 2019-2025, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V. & The NNCodec Authors.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted (subject to the limitations in the disclaimer below) provided that
the following conditions are met:

     * Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.

     * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

     * Neither the name of the copyright holder nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
'''
import sys
import os
from glob import glob
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import setuptools

# --------- Metadata ---------
MIN_PYTHON = (3, 8)
__version__ = '2.1.1'
if sys.version_info < MIN_PYTHON:
    sys.exit(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or later is required.")

# --------- Pybind11 include helper ---------
class get_pybind_include:
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

# --------- C++ source and headers ---------
sources = (
    glob("src/nncodec/extensions/deepCABAC/source/*.cpp") +
    glob("src/nncodec/extensions/deepCABAC/source/Lib/CommonLib/*.cpp") +
    glob("src/nncodec/extensions/deepCABAC/source/Lib/EncLib/*.cpp") +
    glob("src/nncodec/extensions/deepCABAC/source/Lib/DecLib/*.cpp")
)
for source in sources:
    print(source)

include_dirs = [
    get_pybind_include(),
    get_pybind_include(user=True),
    "src/nncodec/extensions/deepCABAC/source",
    "src/nncodec/extensions/deepCABAC/source/Lib",
    "src/nncodec/extensions/deepCABAC/source/Lib/CommonLib",
    "src/nncodec/extensions/deepCABAC/source/Lib/EncLib",
    "src/nncodec/extensions/deepCABAC/source/Lib/DecLib",
]

# --------- Extension ---------
ext_modules = [
    Extension(
        'nncodec.extensions.deepCABAC',
        sources=sources,
        language='c++'  # no include_dirs here, set later
    ),
]

# --------- Compiler flags ---------
def has_flag(compiler, flagname):
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main(int, char**) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True

def cpp_flag(compiler):
    for flag in ['-std=c++17', '-std=c++14', '-std=c++11']:
        if has_flag(compiler, flag):
            return flag
    raise RuntimeError('Unsupported compiler: C++11 or better required.')

class BuildExt(build_ext):
    c_opts = {'msvc': ['/EHsc'], 'unix': []}
    l_opts = {'msvc': [], 'unix': []}

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.14']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == 'unix':
            opts.append(f'-DVERSION_INFO="{self.distribution.get_version()}"')
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append(f'/DVERSION_INFO=\\"{self.distribution.get_version()}\\"')

        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
            ext.include_dirs = include_dirs

        build_ext.build_extensions(self)


# --------- Setup ---------
setup(
    name='nncodec',
    version=__version__,
    author='Paul Haase, Daniel Becking',
    author_email='paul.haase@hhi.fraunhofer.de, daniel.becking@hhi.fraunhofer.de',
    url='https://hhi.fraunhofer.de',
    description='Fraunhofer HHI implementation of the Neural Network Coding (NNC) Standard',
    long_description=open("README.md", encoding="utf-8").read(),
    license='BSD',
    license_files=['LICENSE'],
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    python_requires='>=3.8',
    install_requires=[
        "Click>=7.0",
        "scikit-learn>=0.23.1",
        "tqdm>=4.32.2",
        "h5py>=3.1.0",
        "pybind11>=2.6.2",
        "pandas>=1.0.5",
        "opencv-python>=4.4.0.46",
        "torch>=2",
        "torchvision>=0.15",
        "ptflops>=0.7",
        "torchmetrics>=0.11.4",
        "numpy<2",
        # "sentencepiece>=0.1.99",
        # "wandb>=0.15.3",
        # "matplotlib>=3.7.1",
        # "flwr[simulation]>=1.5",
        # "hydra-core>=1.3.2",
    ],
    setup_requires=['pybind11>=2.6.2'],
    zip_safe=False,
    include_package_data=True,
    package_data={"nncodec.extensions.deepCABAC": ["source/**/*.h"]},
)
