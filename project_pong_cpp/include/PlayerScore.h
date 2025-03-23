//
// Created by Tobenna Udeze on 3/21/25.
//

#ifndef PROJECT_PONG_CPP_PLAYERSCORE_H
#define PROJECT_PONG_CPP_PLAYERSCORE_H
#include <SDL.h>
#include <SDL2/SDL_ttf.h>
#include "Constants.h"
#include "Vec2.h"

class PlayerScore {
public:
    PlayerScore(Vec2 position, SDL_Renderer* renderer, TTF_Font* font);
    ~PlayerScore();
    void Draw();
    SDL_Renderer* renderer;
    TTF_Font* font;
    SDL_Surface* surface{};
    SDL_Texture* texture{};
    SDL_Rect rect{};
};

#endif //PROJECT_PONG_CPP_PLAYERSCORE_H
