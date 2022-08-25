import ctypes
import os
import shutil
import configparser


def agrifield(pInput, pResult, pTemp, DLLPath, DLLName):
    # 将当前目录设置为dll的路径
    os.chdir(DLLPath)
    print(os.getcwd())
    SimDeepDll = ctypes.cdll.LoadLibrary(DLLName)

    SimDeepDll.CreateAgriField(pInput.encode(encoding='UTF-8'),
                               pResult.encode(encoding='UTF-8'),
                               pTemp.encode(encoding='UTF-8'), )


def building(pInput, pResult, pTemp, DLLPath, DLLName):
    # 将当前目录设置为dll的路径
    os.chdir(DLLPath)
    print(os.getcwd())
    SimDeepDll = ctypes.cdll.LoadLibrary(DLLName)

    SimDeepDll.CreateBuilding(pInput.encode(encoding='UTF-8'),
                              pResult.encode(encoding='UTF-8'),
                              pTemp.encode(encoding='UTF-8'), )


def all_element(pInput, pResult, pTemp, DLLPath, DLLName):
    # 将当前目录设置为dll的路径
    os.chdir(DLLPath)
    print(os.getcwd())
    SimDeepDll = ctypes.cdll.LoadLibrary(DLLName)

    SimDeepDll.CreateAllElement(pInput.encode(encoding='UTF-8'),
                                pResult.encode(encoding='UTF-8'),
                                pTemp.encode(encoding='UTF-8'), )
#
# extern "C"  __declspec(dllexport) bool CreateAgriField(const char* pInput, const char* pResult, const char* pTemp);
#
# extern "C"  __declspec(dllexport) bool CreateBuilding(const char* pInput, const char* pResult, const char* pTemp);
#
# extern "C"  __declspec(dllexport) bool CreateAllElement(const char* pInput, const char* pResult, const char* pTemp);


if __name__ == '__main__':

    DLLPath = r"F:\x64\Debug"  # 封装好的dll所在文件夹路径
    DLLName = "SimDeep.dll"

    # 多文件处理
    dir = r'G:\dataset\tianchi\trainData\val\0'  #
    result_path = r'G:\dataset\tianchi\trainData\val\res'  # 输出结果路径
    temp_path = r"G:\dataset\tianchi\trainData\val\temp"  # 临时文件
    file_list = os.listdir(dir)
    for i in file_list:
        if i.endswith('.png'):
            file_path = os.path.join(dir, i)

            agrifield(file_path, result_path, temp_path, DLLPath, DLLName)

    # # 单个文件处理
