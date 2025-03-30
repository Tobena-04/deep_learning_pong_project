//
// Created by Tobenna Udeze on 3/18/25.
//

#include "../include/Ball.h"

Ball::Ball(Vec2 position, Vec2 velocity) :position(position), velocity(velocity)
{
    rect.x = static_cast<int>(position.x);
    rect.y = static_cast<int>(position.y);
    rect.w = BALL_WIDTH;
    rect.h = BALL_HEIGHT;
}

void Ball::Update(float dt)
{
    position += velocity * dt;
}

void Ball::Draw(SDL_Renderer * renderer)
{
    rect.x = static_cast<int>(position.x);
    rect.y = static_cast<int>(position.y);

    SDL_RenderFillRect(renderer, &rect);
}

void Ball::CollideWithPaddle(const Contact &contact)
{
    position.x += contact.penetration;
    velocity.x = -velocity.x;

    if (contact.type == CollisionType::Top){
        velocity.y = -0.75f * BALL_SPEED;
    } else if (contact.type == CollisionType::Bottom){
        velocity.y = 0.75f * BALL_SPEED;
    }
}

void Ball::CollideWithWall(Contact const& contact)
{
    if ((contact.type == CollisionType::Top) || (contact.type == CollisionType::Bottom)){
        position.y += contact.penetration;
        velocity.y = -velocity.y;
    } else if (contact.type == CollisionType::Left){
        position.x = WINDOW_WIDTH/2.0f;
        position.y = WINDOW_HEIGHT/2.0f;
        velocity.x = BALL_SPEED;
        velocity.y = 0.75f * BALL_SPEED;
    } else if (contact.type == CollisionType::Right){
        position.x = WINDOW_WIDTH/2.0f;
        position.y = WINDOW_HEIGHT/2.0f;
        velocity.x = -BALL_SPEED;
        velocity.y = 0.75f * BALL_SPEED;
    }
}