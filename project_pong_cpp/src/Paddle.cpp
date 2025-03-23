//
// Created by Tobenna Udeze on 3/21/25.
//

#include "../include/Paddle.h"

Paddle::Paddle(Vec2 position): position(position)
{
    rect.x = static_cast<int>(position.x);
    rect.y = static_cast<int>(position.y);
    rect.w = PADDLE_WIDTH;
    rect.h = PADDLE_HEIGHT;
}

void Paddle::Draw(SDL_Renderer* renderer)
{
    rect.y = static_cast<int>(position.y);

    SDL_RenderFillRect(renderer, &rect);
}