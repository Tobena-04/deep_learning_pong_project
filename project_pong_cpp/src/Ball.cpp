//
// Created by Tobenna Udeze on 3/18/25.
//

#include "../include/Ball.h"

Ball::Ball(Vec2 position) :position(position)
{
    rect.x = static_cast<int>(position.x);
    rect.y = static_cast<int>(position.y);
    rect.w = BALL_WIDTH;
    rect.h = BALL_HEIGHT;
}

void Ball::Draw(SDL_Renderer * renderer)
{
    rect.x = static_cast<int>(position.x);
    rect.y = static_cast<int>(position.y);

    SDL_RenderFillRect(renderer, &rect);
}