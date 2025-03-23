#include <chrono>
#include <SDL.h>
#include <SDL2/SDL_ttf.h>
#include "include/Ball.h"
#include "include/Paddle.h"
#include "include/PlayerScore.h"
#include "include/Constants.h"

const int WINDOW_WIDTH = 1280;
const int WINDOW_HEIGHT = 720;

int main()
{
    // Initializing SDL components
    SDL_Init(SDL_INIT_VIDEO);
    TTF_Init();

    SDL_Window* window = SDL_CreateWindow("Pong", 0, 0, WINDOW_WIDTH,
                                          WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, 0);

    // Initialize the font
    TTF_Font* scoreFont = TTF_OpenFont("/Users/tobennaudeze/Library/Fonts/DejaVuSansMono.ttf",
                                       40);

    // Create the player score text fields
    PlayerScore playerOneScoreText(Vec2(WINDOW_WIDTH/4.0f, 20.0f), renderer, scoreFont);

    PlayerScore playerTwoScoreText(Vec2(3*WINDOW_WIDTH/4.0f, 20.0f), renderer, scoreFont);

    // Create the ball
    Ball ball(
            Vec2((WINDOW_WIDTH/2.0f) - (BALL_WIDTH/2.0f),
                 (WINDOW_HEIGHT/2.0f)-(BALL_WIDTH/2.0f)));

    // Create the paddles
    Paddle paddleOne(Vec2(50.0f, (WINDOW_HEIGHT/2.0f)-(PADDLE_HEIGHT/2.0f)),
            Vec2(0.0f, 0.0f));

    Paddle paddleTwo(
            Vec2(WINDOW_WIDTH - 50.0f, (WINDOW_HEIGHT/2.0f) - (PADDLE_HEIGHT/2.0f)),
            Vec2(0.0f, 0.0f));

    // Game logic
    {
        enum Buttons
        {
            PaddleOneUp = 0,
            PaddleOneDown,
            PaddleTwoUp,
            PaddleTwoDown,
        };
        const float PADDLE_SPEED = 1.0f;

        bool running = true;
        bool buttons[4] = {};

        float dt = 0.0f;

        // Continue looping and processing events until user exits
        while (running){
            auto startTime = std::chrono::high_resolution_clock ::now();
            SDL_Event event;
            while (SDL_PollEvent(&event)){
                if (event.type == SDL_QUIT){
                    running = false;
                } else if (event.type == SDL_KEYDOWN){
                    if (event.key.keysym.sym == SDLK_ESCAPE){
                        running = false;
                    }
                }
            }

            // Clear the window to black
            SDL_SetRenderDrawColor(renderer, 0x0, 0x0, 0x0, 0xFF);
            SDL_RenderClear(renderer);

            // Set the draw color to white
            SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0xFF, 0xFF);

            // Draw net
            for (int y = 0; y < WINDOW_HEIGHT; ++y)
            {
                if (y%5){
                    SDL_RenderDrawPoint(renderer, WINDOW_WIDTH/2, y);
                }
            }

            // Rendering will happen here

            // Draw the ball
            ball.Draw(renderer);

            // Draw the paddles
            paddleOne.Draw(renderer);
            paddleTwo.Draw(renderer);

            // Display the scores
            playerOneScoreText.Draw();
            playerTwoScoreText.Draw();

            // Present the backbuffer
            SDL_RenderPresent(renderer);

            // Calculate frame time
            auto stopTime = std::chrono::high_resolution_clock::now();
            dt = std::chrono::duration<float, std::chrono::milliseconds::period>
                    (stopTime - startTime).count();
        }
    }

    // Cleanup
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    TTF_CloseFont(scoreFont);
    TTF_Quit();
    SDL_Quit();

    return 0;
}