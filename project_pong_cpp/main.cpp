#include <chrono>
#include <SDL.h>
#include <SDL2/SDL_ttf.h>
#include "include/Ball.h"
#include "include/Composites.h"
#include "include/Constants.h"
#include "include/Paddle.h"
#include "include/PlayerScore.h"

Contact CheckPaddleCollision(Ball const& ball, Paddle const& paddle)
{
    float ballLeft = ball.position.x;
    float ballRight = ball.position.x + BALL_WIDTH;
    float ballTop = ball.position.y;
    float ballBottom = ball.position.y + BALL_HEIGHT;

    float paddleLeft = paddle.position.x;
    float paddleRight = paddle.position.x + PADDLE_WIDTH;
    float paddleTop = paddle.position.y;
    float paddleBottom = paddle.position.y + PADDLE_HEIGHT;

    Contact contact{};

    if (ballLeft >= paddleRight){
        return contact;
    }

    if (ballRight <= paddleLeft){
        return contact;
    }

    if (ballTop >= paddleBottom){
        return contact;
    }

    if (ballBottom <= paddleTop){
        return contact;
    }

    float paddleRangeUpper = paddleBottom - (2.0f * PADDLE_HEIGHT/3.0f);
    float paddleRangeMiddle = paddleBottom - (PADDLE_HEIGHT / 3.0f);

    if (ball.velocity.x < 0){
        // Left paddle
        contact.penetration = paddleRight - ballLeft;
    } else if (ball.velocity.x > 0){
        // Right paddle
        contact.penetration = paddleLeft - ballRight;
    }

    if ((ballBottom > paddleTop) && (ballBottom < paddleRangeUpper)){
        contact.type = CollisionType::Top;
    } else if ((ballBottom > paddleRangeUpper) && (ballBottom < paddleRangeUpper)){
        contact.type = CollisionType::Middle;
    } else{
        contact.type = CollisionType::Bottom;
    }

    return contact;
}

Contact CheckWallCollision(Ball const& ball)
{
    float ballLeft = ball.position.x;
    float ballRight = ball.position.x + BALL_WIDTH;
    float ballTop = ball.position.y;
    float ballBottom = ball.position.y + BALL_HEIGHT;

    Contact contact{};

    if (ballLeft < 0.0f){
        contact.type = CollisionType::Left;
    } else if (ballRight > WINDOW_WIDTH){
        contact.type = CollisionType::Right;
    } else if (ballTop < 0.0f){
        contact.type = CollisionType::Top;
        contact.penetration = -ballTop;
    } else if (ballBottom > WINDOW_HEIGHT){
        contact.type = CollisionType::Bottom;
        contact.penetration = WINDOW_HEIGHT - ballBottom;
    }

    return contact;
}

int main()
{
    // Initializing SDL components
    SDL_Init(SDL_INIT_VIDEO);
    TTF_Init();

    SDL_Window* window = SDL_CreateWindow("Pong", 0, 0, WINDOW_WIDTH,
                                          WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, 0);

    // Initialize the font - TODO: change this line
    TTF_Font* scoreFont = TTF_OpenFont("/Users/tobennaudeze/Library/Fonts/DejaVuSansMono.ttf",
                                       40);

    // Create the player score text fields
    PlayerScore playerOneScoreText(Vec2(WINDOW_WIDTH/4.0f, 20.0f), renderer, scoreFont);

    PlayerScore playerTwoScoreText(Vec2(3*WINDOW_WIDTH/4.0f, 20.0f), renderer, scoreFont);

    // Create the ball
    Ball ball(
            Vec2(WINDOW_WIDTH/2.0f,WINDOW_HEIGHT/2.0f),
            Vec2(BALL_SPEED, 0.0f));

    // Create the paddles
    Paddle paddleOne(Vec2(50.0f, (WINDOW_HEIGHT/2.0f)-(PADDLE_HEIGHT/2.0f)),
            Vec2(0.0f, 0.0f));

    Paddle paddleTwo(
            Vec2(WINDOW_WIDTH - 50.0f, (WINDOW_HEIGHT/2.0f) - (PADDLE_HEIGHT/2.0f)),
            Vec2(0.0f, 0.0f));

    // Game logic
    {
        int playerOneScore = 0;
        int playerTwoScore = 0;

        bool running = true;
        bool buttons[4] = {};

        float dt = 0.0f;

        // Continue looping and processing events until user exits
        while (running){
            auto startTime = std::chrono::high_resolution_clock ::now();
            SDL_Event event;
            while (SDL_PollEvent(&event)){
                if (event.type == SDL_QUIT || playerOneScore == 10 || playerTwoScore == 10){
                    running = false;
                } else if (event.type == SDL_KEYDOWN){
                    if (event.key.keysym.sym == SDLK_ESCAPE){
                        running = false;
                    } else if (event.key.keysym.sym == SDLK_w){
                        buttons[Buttons::PaddleOneUp] = true;
                    } else if (event.key.keysym.sym == SDLK_s){
                        buttons[Buttons::PaddleOneDown] = true;
                    } else if (event.key.keysym.sym == SDLK_UP){
                        buttons[Buttons::PaddleTwoUp] = true;
                    } else if (event.key.keysym.sym == SDLK_DOWN){
                        buttons[Buttons::PaddleTwoDown] = true;
                    }
                } else if (event.type == SDL_KEYUP) {
                    if (event.key.keysym.sym == SDLK_w){
                        buttons[Buttons::PaddleOneUp] = false;
                    } else if (event.key.keysym.sym == SDLK_s){
                        buttons[Buttons::PaddleOneDown] = false;
                    } else if (event.key.keysym.sym == SDLK_UP){
                        buttons[Buttons::PaddleTwoUp] = false;
                    } else if (event.key.keysym.sym == SDLK_DOWN){
                        buttons[Buttons::PaddleTwoDown] = false;
                    }
                }
            }

            // Change paddle velocity on button press
            if (buttons[Buttons::PaddleOneUp]){
                paddleOne.velocity.y = -PADDLE_SPEED;
            } else if (buttons[Buttons::PaddleOneDown]){
                paddleOne.velocity.y = PADDLE_SPEED;
            } else{
                paddleOne.velocity.y = 0.0f;
            }

            if (buttons[Buttons::PaddleTwoUp]){
                paddleTwo.velocity.y = -PADDLE_SPEED;
            } else if (buttons[Buttons::PaddleTwoDown]){
                paddleTwo.velocity.y = PADDLE_SPEED;
            } else{
                paddleTwo.velocity.y = 0.0f;
            }

            // Update paddle positions
            paddleOne.Update(dt);
            paddleTwo.Update(dt);

            // Update ball positions
            ball.Update(dt);

            // Check collisions
           if (Contact contact = CheckPaddleCollision(ball, paddleOne);
           contact.type != CollisionType::None){
               ball.CollideWithPaddle(contact);
           } else if (contact = CheckPaddleCollision(ball, paddleTwo);
           contact.type != CollisionType::None){
               ball.CollideWithPaddle(contact);
           } else if (contact = CheckWallCollision(ball);
           contact.type != CollisionType::None){
               ball.CollideWithWall(contact);

               if (contact.type == CollisionType::Left){
                   ++playerTwoScore;
                   playerTwoScoreText.SetScore(playerTwoScore);
                   
               } else if (contact.type == CollisionType::Right){
                   ++playerOneScore;
                   playerOneScoreText.SetScore(playerOneScore);
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