# Deep Learning Pong Project
By Catherine Reller, Fardeen Bablu, and Tobenna Udeza
This repository contains the learning task for our deep learning network.

## How to Run Pong

### Prerequisites

You will need to install SDL2 and SDL_TTF using Homebrew:

```
brew install sdl2
brew install sdl2_ttf
```

### Configure SDL2 and SDL_TTF

After installation, locate the directory containing the `.h` files for SDL2 and SDL_TTF. The default directory may vary, so you may need to look it up. Below are some example paths:

- **CMAKE_PREFIX_PATH**: `/opt/homebrew` (Modify line 6 in `CMakeLists.txt`)
- **Include Directories**: `/opt/homebrew/cellar/sdl2_ttf/2.24.0/include` (Modify line 15)
- **Target Link Libraries**: `/opt/homebrew/cellar/sdl2_ttf/2.24.0/lib/libSDL2_ttf.dylib` (Modify line 35)

### IDE Considerations

Avoid running this project in VS Code unless your C++ environment is properly set up, as it may cause issues.

### Font Configuration

Ensure the required font is in your Font Book, as they should be stored in that application here: 

`TTF_Font* scoreFont = TTF_OpenFont("/Users/*username*/Library/Fonts/DejaVuSansMono.ttf", 40);`

## Running the Project

Once everything is set up, you should be able to run Pong. Follow these steps:

```
cd project_pong_cpp/build
cmake ..
make
./project_pong_cpp
```

