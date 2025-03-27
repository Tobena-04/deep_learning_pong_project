//
// Created by Tobenna Udeze on 3/21/25.
//

#include <string>
#include "../include/PlayerScore.h"

PlayerScore::PlayerScore(Vec2 position, SDL_Renderer *renderer, TTF_Font *font)
    : renderer(renderer), font(font)
{
    surface = TTF_RenderText_Solid(font, "0", {0xFF, 0xFF, 0xFF, 0xFf});
    texture = SDL_CreateTextureFromSurface(renderer, surface);

    int width, height;
    SDL_QueryTexture(texture, nullptr, nullptr, &width, &height);

    rect.x = static_cast<int>(position.x);
    rect.y = static_cast<int>(position.y);
    rect.w = width;
    rect.h = height;
}

PlayerScore::~PlayerScore()
{
    SDL_FreeSurface(surface);
    SDL_DestroyTexture(texture);
}

void PlayerScore::SetScore(int score)
{
    SDL_FreeSurface(surface);
    SDL_DestroyTexture(texture);

    surface = TTF_RenderText_Solid(font, std::to_string(score).c_str(),
                                   {0xFF, 0xFF, 0xFF, 0xFF});
    texture = SDL_CreateTextureFromSurface(renderer, surface);

    int width, height;
    SDL_QueryTexture(texture, nullptr, nullptr, &width, &height);
    rect.w = width;
    rect.h = height;
}

void PlayerScore::Draw()
{
    SDL_RenderCopy(renderer, texture, nullptr, &rect);
}