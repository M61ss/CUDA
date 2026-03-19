# Learning material

Inside folders there is a README and there are some notes written in code comments. Detailed information are available on the NVIDIA official documentation site "[NVIDIA CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-model)".

Cool CUDA cheat sheets: https://kdm.icm.edu.pl/Tutorials/GPU-intro/introduction.en/.

In this repository there is also `basic-sample`, an auto-generated Visual Studio CUDA project that I think could be useful as a good starting point for refresh your memory or keep a reference about fundamental functions and conventions usually used with CUDA.

# Run programs

## Windows + VSCode

1. Download and install Visual Studio (needed for C++ compiler and libraries);
2. Download and install CUDA Toolkit for Windows from official website;
3. Download and install NSight Visual Studio Code Edition from VSCode Marketplace;
4. Run program from PowerShell using:
   ```powershell
   nvcc your_program.cu -o your_program.exe -ccbin path\to\cl.exe
   ```
   For Visual Studio 2022 Enterprise the path to cl.exe is: `"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64"`.
   
> [!TIP]
>
> You can avoid specify every time `-ccbin` adding the path to cl.exe to PATH environment variable. In order to do this, press `Win+R`, type `sysdm.cpl`, go to "Advanced", click on "Environment Variables..." and edit `Path`.

# Profiling

## Windows + NSight CLI

1. If not installed, download and install NSight System (it should be bundled with CUDA Toolkit). To check if it is installed, search "NSight Systems" folder in `"C:\Program Files\NVIDIA Corporation\"`;
2. Add nsys.exe to the path (for instance its path could be `"C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.5.1\target-windows-x64\"`);
3. Profile your program in an ADMINISTRATOR PowerShell session:
   ```powershell
   nsys profile -t cuda --stats=true -o report_name .\your_program.exe
   ```
