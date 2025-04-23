//
// Created by Tobenna Udeze on 3/23/25.
//

#ifndef PROJECT_PONG_CPP_CONSTANTS_H
#define PROJECT_PONG_CPP_CONSTANTS_H

#include <cfloat>

// Constants used across the project
inline constexpr int WINDOW_WIDTH = 1280;
inline constexpr int WINDOW_HEIGHT = 720;
inline constexpr int BALL_WIDTH = 15;
inline constexpr int BALL_HEIGHT = 15;
inline constexpr int PADDLE_WIDTH = 10;
inline constexpr int PADDLE_HEIGHT = 100;


inline constexpr float PADDLE_SPEED = FLT_MAX;
inline constexpr float BALL_SPEED = FLT_MAX;

// Faster version for training
//inline constexpr float PADDLE_SPEED = 100.0f;
//inline constexpr float BALL_SPEED = 120.0f;

#endif