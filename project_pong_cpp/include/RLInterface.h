//
// Created by Fardeen Bablu on 4/15/25.
//

#ifndef PROJECT_PONG_CPP_RLINTERFACE_H
#define PROJECT_PONG_CPP_RLINTERFACE_H

#include <fstream>
#include <vector>
#include <string>
#include "Constants.h"
#include "Ball.h"
#include "Paddle.h"

// Structure to hold the game state for RL
struct GameState {
    float ball_x;
    float ball_y;
    float ball_vel_x;
    float ball_vel_y;
    float left_paddle_y;
    float right_paddle_y;
    int left_score;
    int right_score;
};

class RLInterface {
public:
    RLInterface(const std::string& state_file = "game_state.txt",
                const std::string& action_file = "agent_action.txt",
                const std::string& done_file = "game_done.txt")
        : state_filename(state_file), action_filename(action_file), done_filename(done_file) {

        // Create an empty action file
        std::ofstream action_file_stream(action_filename);
        action_file_stream << "0" << std::endl;  // 0 means no action
        action_file_stream.close();

        // Create an empty done file
        std::ofstream done_file_stream(done_filename);
        done_file_stream << "0" << std::endl;  // 0 means game is not done
        done_file_stream.close();
    }

    void updateGameState(const Ball& ball, const Paddle& left_paddle,
                         const Paddle& right_paddle, int left_score, int right_score) {
        // Prepare game state
        GameState state;
        state.ball_x = ball.position.x / WINDOW_WIDTH;  // Normalize to 0-1
        state.ball_y = ball.position.y / WINDOW_HEIGHT;
        state.ball_vel_x = ball.velocity.x / BALL_SPEED;
        state.ball_vel_y = ball.velocity.y / BALL_SPEED;
        state.left_paddle_y = left_paddle.position.y / WINDOW_HEIGHT;
        state.right_paddle_y = right_paddle.position.y / WINDOW_HEIGHT;
        state.left_score = left_score;
        state.right_score = right_score;

        // Write to file
        std::ofstream state_file(state_filename);
        state_file << state.ball_x << ","
                   << state.ball_y << ","
                   << state.ball_vel_x << ","
                   << state.ball_vel_y << ","
                   << state.left_paddle_y << ","
                   << state.right_paddle_y << ","
                   << state.left_score << ","
                   << state.right_score << std::endl;
        state_file.close();
    }

    int getAgentAction() {
        // Read action from file
        std::ifstream action_file(action_filename);
        int action = 0;
        action_file >> action;
        action_file.close();
        return action;  // 0: no move, 1: up, 2: down
    }

    void setGameDone(bool done) {
        std::ofstream done_file(done_filename);
        done_file << (done ? "1" : "0") << std::endl;
        done_file.close();
    }

private:
    std::string state_filename;
    std::string action_filename;
    std::string done_filename;
};

#endif //PROJECT_PONG_CPP_RLINTERFACE_H
