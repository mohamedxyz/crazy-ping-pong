# TODO: add comments
import turtle
import random
window = turtle.Screen()
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
window.bgcolor("black")
window.setup(SCREEN_WIDTH, SCREEN_HEIGHT)
window.tracer(0)
colors = ["pink", "magenta", "orange", "green"]


class Paddle:
    def __init__(self, X, Y, color):
        self.paddle = turtle.Turtle()
        self.paddle.speed(0)
        self.paddle.shape("square")
        self.paddle.color(color)
        self.paddle.penup()
        self.paddle.goto(X, Y)
        self.paddle.shapesize(stretch_wid=5, stretch_len=1)

    def up(self):
        y = self.paddle.ycor()
        y += 20
        self.paddle.sety(y)

    def down(self):
        y = self.paddle.ycor()
        y -= 20
        self.paddle.sety(y)


paddle1 = Paddle(-350, 0, "blue")
paddle2 = Paddle(350, 0, "red")


class Ball:
    def __init__(self, X, Y, balldx, balldy):
        # ball attributes
        self.ball = turtle.Turtle()
        self.ball.speed(0)
        self.ball.shape("circle")
        self.ball.color("yellow")
        self.ball.penup()
        self.ball.goto(X, Y)
        self.balldx = balldx
        self.balldy = balldy
    def reset(self):
        self.ball.goto(0, 0)
        self.balldx *= -1
        self.ball.color(random.choice(colors))
    def move(self):
        # make the ball move
        self.ball.setx(self.ball.xcor() + self.balldx)
        self.ball.sety(self.ball.ycor() + self.balldy)
        # check if ball collide with borders
        if self.ball.xcor() > (SCREEN_WIDTH/2):
            self.reset()
        if self.ball.xcor() < (SCREEN_WIDTH/-2):
            self.reset()
        if self.ball.ycor() > (SCREEN_HEIGHT/2):
            self.ball.sety(290)
            self.balldy *= -1
        if self.ball.ycor() < (SCREEN_HEIGHT/-2):
            self.ball.sety(-290)
            self.balldy *= -1

    # def ball_collision_paddle(self, paddle):
    #     if (self.ball.xcor() > 380) and (self.ball.xcor() > paddle.xcor() and self.ball.ycor() > paddle.ycor()):
    #         self.ball.goto(0, 0)


balldx = 0.5
balldy = 0.5
ball_1 = Ball(0, 0, 0.5, 0.5)
ball_2 = Ball(0, 0, 0.7, 0.7)
window.listen()
window.onkeypress(paddle1.up, "w")
window.onkeypress(paddle1.down, "s")
window.onkeypress(paddle2.up, "Up")
window.onkeypress(paddle2.down, "Down")
while True:
    window.update()
    ball_1.move()
    ball_2.move()
    # ball_1.ball_collision_paddle(paddle1)
    # ball_1.ball_collision_paddle(paddle2)
    # ball_2.ball_collision_paddle(paddle1)
    # ball_2.ball_collision_paddle(paddle2)
