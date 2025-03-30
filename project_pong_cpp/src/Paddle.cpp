//
// Created by Tobenna Udeze on 3/21/25.
//

#include "../include/Paddle.h"

Paddle::Paddle(Vec2 position, Vec2 velocity): position(position), velocity(velocity)
{
    rect.x = static_cast<int>(position.x);
    rect.y = static_cast<int>(position.y);
    rect.w = PADDLE_WIDTH;
    rect.h = PADDLE_HEIGHT;
}

void Paddle::Update(float dt)
{
    position += velocity * dt;

    if (position.y < 0){
        // Restrict to top of the screen
        position.y = 0;
    } else if (position.y > (WINDOW_HEIGHT - PADDLE_HEIGHT)){
        // Restrict to bottom of the screen
        position.y = WINDOW_HEIGHT - PADDLE_HEIGHT;
    }
}

void Paddle::Draw(SDL_Renderer* renderer)
{
    rect.y = static_cast<int>(position.y);

    SDL_RenderFillRect(renderer, &rect);
}