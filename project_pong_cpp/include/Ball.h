//
// Created by Tobenna Udeze on 3/18/25.
//

#ifndef PROJECT_PONG_CPP_BALL_H
#define PROJECT_PONG_CPP_BALL_H

#include "Vec2.h"
#include "Constants.h"
#include "Composites.h"
#include <SDL.h>

class Ball{
public:
    Ball(Vec2 position, Vec2 velocity);
    void Draw(SDL_Renderer* renderer);
    void Update(float dt);
    void CollideWithPaddle(Contact const& contact);
    void CollideWithWall(Contact const& contact);
    Vec2 position;
    Vec2 velocity;
    SDL_Rect rect{};
};

#endif //PROJECT_PONG_CPP_BALL_H
