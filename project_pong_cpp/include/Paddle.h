//
// Created by Tobenna Udeze on 3/21/25.
//

#ifndef PROJECT_PONG_CPP_PADDLE_H
#define PROJECT_PONG_CPP_PADDLE_H
#include "Vec2.h"
#include <SDL.h>

const int PADDLE_WIDTH = 10;
const int PADDLE_HEIGHT = 100;

class Paddle{
public:
    Paddle(Vec2 position);
    void Draw(SDL_Renderer* renderer);
    Vec2 position;
    SDL_Rect rect{};
};

#endif //PROJECT_PONG_CPP_PADDLE_H
