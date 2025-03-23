//
// Created by Tobenna Udeze on 3/18/25.
//

#ifndef PROJECT_PONG_CPP_VEC2_H
#define PROJECT_PONG_CPP_VEC2_H

#include "Constants.h"

class Vec2 {
public:
    Vec2();
    Vec2(float x, float y);
    Vec2 operator+(Vec2 const& rhs) const;
    Vec2& operator+=(Vec2 const& rhs);
    Vec2 operator*(float rhs) const;
    float x, y;
};

#endif //PROJECT_PONG_CPP_VEC2_H
