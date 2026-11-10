# #
# # import ctypes
# # from pathlib import Path
# # import transcribe_cpp
# # from crispasr import _binding
# #
# # dll_path = Path(_binding._find_lib()).resolve()
# #
# # print(f"CrispASR DLL: {dll_path}")
# # print(f"DLL exists: {dll_path.is_file()}")
# #
# # try:
# #     library = ctypes.CDLL(str(dll_path))
# #     print("CrispASR DLL loaded successfully")
# # except OSError as error:
# #     print(f"DLL loading failed: {error}")
#
#
# import ctypes
# from pathlib import Path
#
# # 先載入 CrispASR
# from crispasr import _binding
#
# dll_path = Path(_binding._find_lib()).resolve()
# crisp_library = ctypes.CDLL(str(dll_path))
#
# print("CrispASR DLL loaded successfully")
#
# # 再載入 transcribe.cpp
# import transcribe_cpp
#
# print("transcribe.cpp imported successfully")

import ctypes
from ctypes import wintypes

import transcribe_cpp

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
kernel32.GetModuleHandleW.restype = wintypes.HMODULE

kernel32.GetModuleFileNameW.argtypes = [
    wintypes.HMODULE,
    wintypes.LPWSTR,
    wintypes.DWORD,
]
kernel32.GetModuleFileNameW.restype = wintypes.DWORD

dll_names = [
    "ggml.dll",
    "ggml-base.dll",
    "ggml-cpu.dll",
    "ggml-vulkan.dll",
]

for dll_name in dll_names:
    handle = kernel32.GetModuleHandleW(dll_name)

    if not handle:
        print(f"{dll_name}: Not loaded")
        continue

    path_buffer = ctypes.create_unicode_buffer(32768)

    result = kernel32.GetModuleFileNameW(
        handle,
        path_buffer,
        len(path_buffer),
    )

    if result:
        print(f"{dll_name}: {path_buffer.value}")