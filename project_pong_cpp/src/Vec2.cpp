//
// Created by Tobenna Udeze on 3/18/25.
//

#include "../include/Vec2.h"

Vec2::Vec2(): x(0.0f), y(0.0f)
{}

Vec2::Vec2(float x, float y): x(x), y(y)
{}

Vec2 Vec2::operator+(Vec2 const& rhs) const 
{
    return Vec2(x + rhs.x, y + rhs.y);
}

Vec2& Vec2::operator+=(Vec2 const& rhs)
{
    x+=rhs.x;
    y+=rhs.y;
    return *this;
}

Vec2 Vec2::operator*(float rhs) const
{
    return Vec2(rhs * x, rhs * y);
}