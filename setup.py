from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# ops = CUDAExtension(
#     name='tree_learn.ops.ops',
#     sources=[
#         'tree_learn/ops/src/softgroup_api.cpp', 'tree_learn/ops/src/softgroup_ops.cpp',
#         'tree_learn/ops/src/cuda.cu'
#     ],
#     extra_compile_args={
#         'cxx': ['-g'],
#         'nvcc': ['-O2']
#     })


if __name__ == '__main__':
    setup(
        name='tree_learn',
        packages=["tree_learn"],
        
       # package_data={'tree_learn.ops': ['*/*.so']},
        #cmdclass={'build_ext': BuildExtension},
        #ext_modules=[ops],
        )
