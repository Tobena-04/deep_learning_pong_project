cmake_minimum_required(VERSION 3.28.1)
project(project_pong_cpp)

set(CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")

find_package(SDL2 REQUIRED)
find_package(SDL2_ttf REQUIRED)   
find_package(SDL2_image REQUIRED)
include_directories(
        ${SDL2_INCLUDE_DIRS}
        ${SDL2_TTF_INCLUDE_DIRS}
        ${SDL2_IMAGE_INCLUDE_DIRS}
        /opt/homebrew/cellar/sdl2_ttf/2.24.0/include
)

add_executable(project_pong_cpp main.cpp
        src/Ball.cpp
        src/Vec2.cpp
        include/Vec2.h
        include/Ball.h
        include/Paddle.h
        src/Paddle.cpp
        src/PlayerScore.cpp
        include/PlayerScore.h
        include/Constants.h
        include/Composites.h
        include/RLInterface.h
)

target_link_libraries(project_pong_cpp
        ${SDL2_LIBRARIES}
        ${SDL2_TTF_LIBRARIES}
        ${SDL2_IMAGE_LIBRARIES}
        "/opt/homebrew/cellar/sdl2_ttf/2.24.0/lib/libSDL2_ttf.dylib" # make changes here
)

set_property(TARGET project_pong_cpp PROPERTY CXX_STANDARD 20)