//
// Created by Tobenna Udeze on 3/25/25.
//

#ifndef PROJECT_PONG_CPP_COMPOSITES_H
#define PROJECT_PONG_CPP_COMPOSITES_H

enum Buttons
{
    PaddleOneUp = 0,
    PaddleOneDown,
    PaddleTwoUp,
    PaddleTwoDown,
};

enum class CollisionType
{
    None,
    Top,
    Middle,
    Bottom,
    Left,
    Right
};

struct Contact
{
    CollisionType type;
    float penetration;
};


#endif //PROJECT_PONG_CPP_COMPOSITES_H
