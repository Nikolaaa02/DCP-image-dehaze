
cmake_minimum_required(VERSION 3.16)

project(HazeRemoval_BitAnalysis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    imgcodecs
    highgui
)

add_executable(HazeRemoval_BitAnalysis
    main.cpp
    hazeremoval.cpp
    guidedfilter.cpp
)

target_include_directories(HazeRemoval_BitAnalysis PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(HazeRemoval_BitAnalysis PRIVATE
    ${OpenCV_LIBS}
)
