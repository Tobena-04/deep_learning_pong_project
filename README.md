# deep learning pong project
This repo contains the learning task for our deep learning network. 

## how to run pong
you will need to install SDL2 and SDL_TTF using homebrew

```commandline
brew install SDL2

brew install SDL2_TTF
```
when you have installed it, look for the directory of the ***.h*** files for SDL and SDL_TTF. You 
can google the default directory for this, but it may be different. Mine looked like this:

> "/opt/homebrew" for Appending CMAKE_PREFIX_PATH (line 6 in CMakeList.txt)
> "/opt/homebrew/cellar/sdl2_ttf/2.24.0/include" include directories (line 15)
> "/opt/homebrew/cellar/sdl2_ttf/2.24.0/lib/libSDL2_ttf.dylib" target link libraries (line 35)

I wouldn't advise running this project on vs code unless you're c++ environment is properly set up 
because it can give you a tought time.

lastly, you want to make sure the fonts are accessible because they are saved to your machine on a 
specific directory. I could put a duplicate in here, but I am lazy

```c++
TTF_Font* scoreFont = TTF_OpenFont("/Users/*username*/Library/Fonts/DejaVuSansMono.ttf",
                                       40);
```

change line 97 (it has been marked as TODO) to your own path. this is for mac users.

when this is done, you should be able to run our pong!
=======
# deep_learning_pong_project

This repo contains the learning task for our deep learning network.