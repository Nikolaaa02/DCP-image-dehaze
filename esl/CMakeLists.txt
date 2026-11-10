
cmake_minimum_required(VERSION 3.16)

project(DCP_HazeRemoval LANGUAGES CXX)

# Koristimo C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Pronalaženje OpenCV biblioteke
find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    imgcodecs
    highgui
)

# Kreiranje izvršnog programa
add_executable(HazeRemoval
    main.cpp
    hazeremoval.cpp
    guidedfilter.cpp
)

# Putanje do OpenCV zaglavlja
target_include_directories(HazeRemoval PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

# Povezivanje sa OpenCV bibliotekama
target_link_libraries(HazeRemoval PRIVATE
    ${OpenCV_LIBS}
)
