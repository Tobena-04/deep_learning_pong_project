//
// Created by Tobenna Udeze on 3/21/25.
//

#ifndef PROJECT_PONG_CPP_PADDLE_H
#define PROJECT_PONG_CPP_PADDLE_H
#include "Vec2.h"
#include "Constants.h"
#include <SDL.h>

class Paddle{
public:
    Paddle(Vec2 position, Vec2 velocity);
    void Draw(SDL_Renderer* renderer);
    void Update(float dt);
    Vec2 position;
    Vec2 velocity;
    SDL_Rect rect{};
};

#endif //PROJECT_PONG_CPP_PADDLE_H
