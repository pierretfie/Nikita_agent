#!/usr/bin/env python3

from cffi import FFI
import os

ffi = FFI()
lib = ffi.dlopen("./libllama.so")

# ========== SIMPLE LOADER EXAMPLE ==========

print("[+] Llama loaded successfully (native build)")
print("[+] Ready for inference tasks")

# You will extend this later to actually query the model
