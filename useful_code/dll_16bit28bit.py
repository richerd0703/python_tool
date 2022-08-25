import ctypes
import os
import shutil
import configparser
import os


def use_dll(pInput, pResult, DLLPath, DLLName):
    # 将当前目录设置为dll的路径
    os.chdir(DLLPath)
    print(os.getcwd())
    SimSample = ctypes.cdll.LoadLibrary(DLLName)

    SimSample.gDosStrech16To8(pInput.encode(encoding='UTF-8'), pResult.encode(encoding='UTF-8'))


# bool gDosStrech16To8(const char* pInput, const char* pStrech)
# extern "C"  __declspec(dllexport) bool CreateAgriField(const char* pInput, const char* pResult, const char* pTemp);
#
# extern "C"  __declspec(dllexport) bool CreateBuilding(const char* pInput, const char* pResult, const char* pTemp);
#
# extern "C"  __declspec(dllexport) bool CreateAllElement(const char* pInput, const char* pResult, const char* pTemp);
if __name__ == '__main__':

    # DLLPath = r"F:\wu_dll\x64\Debug"  # 封装好的dll所在文件夹路径
    DLLPath = r"E:\Download\wechart\WeChat Files\wxid_1g7hwiy04a0a21\FileStorage\File\2022-07\x64\Release"  # 封装好的dll所在文件夹路径
    DLLName = "SimSample.dll"

    # 多文件处理
    dir = r'F:\he\data\samples-高二\sz-4bands\215004_d5c6cf9408fe11e9974864006a843cd4'  #
    result_path = r'F:\he\data\zjw'  # 输出结果路径
    temp_path = r"G:\dataset\2021-9-3\res\test_extract\Temp"  # 临时文件
    file_list = os.listdir(dir)
    for i in file_list:
        if i.endswith('.tif'):
            file_path = os.path.join(dir, i)
            use_dll(file_path, result_path, DLLPath, DLLName)

    # # 单个文件处理
