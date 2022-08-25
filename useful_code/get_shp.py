
import ctypes
import os
import shutil
import configparser
import os


def agrifield(pInput, pResult, pTemp, DLLPath, DLLName):
    # 将当前目录设置为dll的路径
    os.chdir(DLLPath)
    print(os.getcwd())
    SimDeepDll = ctypes.cdll.LoadLibrary(DLLName)

    SimDeepDll.CreateAgriField(pInput.encode(encoding='UTF-8'),
                               pResult.encode(encoding='UTF-8'),
                               pTemp.encode(encoding='UTF-8'))


def building(pInput, pResult, pTemp, DLLPath, DLLName):
    # 将当前目录设置为dll的路径
    os.chdir(DLLPath)
    print(os.getcwd())
    SimDeepDll = ctypes.cdll.LoadLibrary(DLLName)

    SimDeepDll.CreateBuilding(pInput.encode(encoding='UTF-8'),
                              pResult.encode(encoding='UTF-8'),
                              pTemp.encode(encoding='UTF-8'))


def all_element(pInput, pResult, pTemp, DLLPath, DLLName):
    # 将当前目录设置为dll的路径
    os.chdir(DLLPath)
    print(os.getcwd())
    SimDeepDll = ctypes.cdll.LoadLibrary(DLLName)

    SimDeepDll.CreateAllElement(pInput.encode(encoding='UTF-8'),
                                pResult.encode(encoding='UTF-8'),
                                pTemp.encode(encoding='UTF-8'))


# extern "C"  __declspec(dllexport) bool CreateAgriField(const char* pInput, const char* pResult, const char* pTemp);
#
# extern "C"  __declspec(dllexport) bool CreateBuilding(const char* pInput, const char* pResult, const char* pTemp);
#
# extern "C"  __declspec(dllexport) bool CreateAllElement(const char* pInput, const char* pResult, const char* pTemp);
if __name__ == '__main__':


    # DLLPath = r"F:\wu_dll\x64\Debug"  # 封装好的dll所在文件夹路径
    DLLPath = r"F:\x64\Release"  # 封装好的dll所在文件夹路径
    DLLName = "SimDeep.dll"

    # 多文件处理
    # dir = r'G:\temp\image'  #
    # result_path = r'G:\temp\shp'  # 输出结果路径
    # temp_path = r"G:\temp\temp"  # 临时文件
    # file_list = os.listdir(dir)
    # for i in file_list:
    #     if i.endswith('.tif'):
    #         file_path = os.path.join(dir, i)
    #         agrifield(file_path, result_path, temp_path, DLLPath, DLLName)

    # 单个文件处理
    file_path = r"G:\temp\image\安吉县_clip1.tif"
    result_path = r'G:\temp\shp'
    temp_path = r"G:\temp\temp"
    # shpfile = file_name.split(".")[0] + "_RetGEO.shp"
    # shpfile = file_name.split(".")[0] + "_Ret.shp"

    building(file_path, result_path, temp_path, DLLPath, DLLName)


