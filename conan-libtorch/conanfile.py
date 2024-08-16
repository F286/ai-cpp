from conan import ConanFile
from conan.tools.files import get, copy
import os

class LibTorchConan(ConanFile):
    name = "libtorch"
    version = "1.9.0"
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "with_cuda": [True, False]}
    default_options = {"shared": True, "with_cuda": False}
    
    def build(self):
        os_name = str(self.settings.os)
        cuda_suffix = "+cu111" if self.options.with_cuda else "+cpu"
        
        urls = {
            "Windows": f"https://download.pytorch.org/libtorch/{'cu111' if self.options.with_cuda else 'cpu'}/libtorch-win-shared-with-deps-{self.version}{cuda_suffix}.zip",
            "Linux": f"https://download.pytorch.org/libtorch/{'cu111' if self.options.with_cuda else 'cpu'}/libtorch-cxx11-abi-shared-with-deps-{self.version}{cuda_suffix}.zip",
            "Macos": f"https://download.pytorch.org/libtorch/cpu/libtorch-macos-{self.version}.zip"  # MacOS doesn't have CUDA version
        }
        
        if os_name not in urls:
            raise Exception(f"OS {os_name} is not supported")
        
        url = urls[os_name]
        get(self, url, strip_root=True)

    def package(self):
        copy(self, "*.h", src=os.path.join(self.build_folder, "include"), dst=os.path.join(self.package_folder, "include"))
        copy(self, "*.lib", src=os.path.join(self.build_folder, "lib"), dst=os.path.join(self.package_folder, "lib"), keep_path=False)
        copy(self, "*.dll", src=os.path.join(self.build_folder, "lib"), dst=os.path.join(self.package_folder, "bin"), keep_path=False)
        copy(self, "*.so", src=os.path.join(self.build_folder, "lib"), dst=os.path.join(self.package_folder, "lib"), keep_path=False)
        copy(self, "*.dylib", src=os.path.join(self.build_folder, "lib"), dst=os.path.join(self.package_folder, "lib"), keep_path=False)

    def package_info(self):
        self.cpp_info.libs = ["torch", "c10"]
        if self.options.with_cuda:
            self.cpp_info.libs.extend(["torch_cuda", "c10_cuda"])
        self.cpp_info.includedirs = ["include", "include/torch/csrc/api/include"]