//
// Created by Tobenna Udeze on 3/18/25.
//

#ifndef PROJECT_PONG_CPP_BALL_H
#define PROJECT_PONG_CPP_BALL_H

#include "Vec2.h"
#include "Constants.h"
#include <SDL.h>

const int BALL_WIDTH = 15;
const int BALL_HEIGHT = 15;

class Ball{
public:
    Ball(Vec2 position);
    void Draw(SDL_Renderer* renderer);
    Vec2 position;
    SDL_Rect rect{};
};

#endif //PROJECT_PONG_CPP_BALL_H
