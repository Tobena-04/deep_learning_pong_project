import turtle

# ball
ball = turtle.Turtle()
ball.shape("circle")
ball.color("white")

ball.penup()
ball.goto(0, 0)
ball.dx = 3
ball.dy = 3

# paddles
paddle1 = turtle.Turtle()
paddle1.shape("square")
paddle1.color("white")
paddle1.shapesize(stretch_wid=5, stretch_len=1)
paddle1.penup()
paddle1.goto(-350, 0)
paddle1.dy = 0

paddle2 = turtle.Turtle()
paddle2.shape("square")
paddle2.color("white")
paddle2.shapesize(stretch_wid=5, stretch_len=1)
paddle2.penup()
paddle2.goto(350, 0)
paddle2.dy = 0

paddle1.sety(paddle1.ycor() + paddle1.dy)
paddle2.sety(paddle2.ycor() + paddle2.dy)
ball.setx(ball.xcor() + ball.dx)
ball.sety(ball.ycor() + ball.dy)

# background
turtle.setup(400, 300)
turtle.bgcolor("black")
# border
def draw_border():
    border = turtle.Turtle()
    border.speed(0)
    border.color("white")
    border.penup
    border.goto(-390, 290)
    border.pendown()
    border.pensize(3)

    for _ in range(2):
        border.forward(780)
        border.right(90)
        border.forward(580)
        border.right(90)
    border.hideturtle

# Function to move paddle1 up
def paddle1_up():
    paddle1.dy += 10

# Function to move paddle1 down
def paddle1_down():
    paddle1.dy -= 10

# Function to move paddle2 up
def paddle2_up():
    paddle2.dy += 10

# Function to move paddle2 down
def paddle2_down():
    paddle2.dy -= 10