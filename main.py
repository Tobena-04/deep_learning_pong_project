import turtle
from elements import ball, paddle1, paddle2, paddle1_down, paddle1_up, paddle2_down, paddle2_up, draw_border

game_over = False
winner = None
points = {
    "player1": 0,
    "player2": 0
}

game_rules = {
    "max_points": 3,
    "ball_speed": 10
}

score_display = turtle.Turtle()
score_display.color("white")
score_display.penup()
score_display.hideturtle()
score_display.goto(0, 300)
score_display.write("Player 1: 0  Player 2: 0", align="center", font=("Arial", 24, "normal"))

draw_border()

# Main game loop
def update():
    global game_over, winner, points

    if not game_over:
        # Move paddles
        paddle1.sety(paddle1.ycor() + paddle1.dy)
        paddle2.sety(paddle2.ycor() + paddle2.dy)

        # Move ball
        ball.setx(ball.xcor() + ball.dx)
        ball.sety(ball.ycor() + ball.dy)

        # Check for ball collision with paddles
        if (ball.xcor() > 340 and ball.xcor() < 350) and (ball.ycor() < paddle2.ycor() + 50 and ball.ycor() > paddle2.ycor() - 50):
            ball.setx(340)
            ball.dx *= -1
        elif (ball.xcor() < -340 and ball.xcor() > -350) and (ball.ycor() < paddle1.ycor() + 50 and ball.ycor() > paddle1.ycor() - 50):
            ball.setx(-340)
            ball.dx *= -1

        # Check for ball going off screen
        if ball.xcor() > 390:
            ball.goto(0, 0)
            ball.dx *= -1
            points["player1"] += 1
            print(f"player 1 {points["player1"]}")
        elif ball.xcor() < -390:
            ball.goto(0, 0)
            ball.dx *= -1
            points["player2"] += 1
            print(f"player 2 {points["player2"]}")

        # Check for ball colliding with top or bottom of screen
        if ball.ycor() > 290:
            ball.sety(290)
            ball.dy *= -1
        elif ball.ycor() < -290:
            ball.sety(-290)
            ball.dy *= -1

        # Check scoring (your existing scoring code here)
        if ball.xcor() > 390:
            ball.goto(0, 0)
            ball.dx *= -1
            points["player1"] += 1
        elif ball.xcor() < -390:
            ball.goto(0, 0)
            ball.dx *= -1
            points["player2"] += 1

            # Update score display
            score_display.clear()
            score_display.write("Player 1: {}  Player 2: {}".format(points["player1"], points["player2"]), align="center", font=("Arial", 24, "normal"))

        # Check game over
        if points["player1"] == game_rules["max_points"] or points["player2"] == game_rules["max_points"]:
            game_over = True
            game_over_display = turtle.Turtle()
            game_over_display.color("white")
            game_over_display.penup()
            game_over_display.hideturtle()
            game_over_display.goto(0, 0)
            winner = "player1" if points["player1"] == game_rules["max_points"] else "player2"
            game_over_display.write("Game Over! {} wins!".format(winner), align="center", font=("Arial", 36, "normal"))
        else:
            turtle.ontimer(update, 10)  # Recursively call update() every 10ms

turtle.listen()
turtle.onkeypress(paddle1_up, "w")
turtle.onkeypress(paddle1_down, "s")
turtle.onkeypress(paddle2_up, "Up")
turtle.onkeypress(paddle2_down, "Down")
turtle.onkeypress(exit, "e")

# Start the game
update()
turtle.mainloop() 


# TODO:
"""
define upper and lower boundary 
make player scores be outside boundary
prevent paddles from crossing boarder
prevent paddle from continuously moving
update scores
"""