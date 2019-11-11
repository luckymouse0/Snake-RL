#Simple snake game to use in a DQN and reinforcement learning

import pygame
import random
import math

#Const variables

#Size of the game window
WINDOW_WIDTH = 360
WINDOW_HEIGHT = 360

#Food const
FOOD_SIZE = 10
FOOD_INCREASE_LENGTH = 1

#Snake const
SNAKE_SPEED = 10
SNAKE_WIDTH = 10
SNAKE_HEIGHT = 10
SNAKE_INITIAL_LENGTH = 2

#RGB colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

#Game class
class SnakeGame:

    #Public variables
    Score = 0
    FoodPosition = [-100,-100]
    SnakeList = []  #snake body parts positions
    SnakeDirection = random.randint(1,4)  #randomized starting direction    
    Screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))  #initialize the Screen width and height


    def __init__(self):
        pygame.font.init()
        
        #Fill the snake position list for the first time
        for i in range(SNAKE_INITIAL_LENGTH):
            x = WINDOW_WIDTH/2
            y = WINDOW_HEIGHT/2
            if (self.SnakeDirection == 1):
                self.SnakeList.append([x, y + (i * SNAKE_HEIGHT)])
            if (self.SnakeDirection == 2):
                self.SnakeList.append([x, y - (i * SNAKE_HEIGHT)])
            if (self.SnakeDirection == 3):
                self.SnakeList.append([x + (i * SNAKE_WIDTH), y])
            if (self.SnakeDirection == 4):
                self.SnakeList.append([x - (i * SNAKE_WIDTH), y])
        
    def newGame(self):
        self.Score = 0
        self.FoodPosition = [-100,-100]
        self.SnakeList.clear()
        self.SnakeDirection = random.randint(1,4)
        
        for i in range(SNAKE_INITIAL_LENGTH):
            x = WINDOW_WIDTH/2
            y = WINDOW_HEIGHT/2
            if (self.SnakeDirection == 1):
                self.SnakeList.append([x, y + (i * SNAKE_HEIGHT)])
            if (self.SnakeDirection == 2):
                self.SnakeList.append([x, y - (i * SNAKE_HEIGHT)])
            if (self.SnakeDirection == 3):
                self.SnakeList.append([x + (i * SNAKE_WIDTH), y])
            if (self.SnakeDirection == 4):
                self.SnakeList.append([x - (i * SNAKE_WIDTH), y])
   
    def update(self, action):    
        #Food
        if(self.FoodPosition[0] == -100):   #drawing new food if there is none
            self.FoodPosition[0] = random.randint(0, WINDOW_WIDTH - FOOD_SIZE)
            self.FoodPosition[1] = random.randint(0, WINDOW_HEIGHT - FOOD_SIZE)

        #Snake
        
        #Update the direction while checking for valid moves
        if (action[1] == 1):  #up
            if(self.SnakeDirection == 3 or self.SnakeDirection == 4):
                self.SnakeDirection = 1
        if (action[2] == 1):  #down
            if(self.SnakeDirection == 3 or self.SnakeDirection == 4):
                self.SnakeDirection = 2
        if (action[3] == 1):  #left
            if(self.SnakeDirection == 1 or self.SnakeDirection == 2):
                self.SnakeDirection = 3
        if (action[4] == 1):  #right
            if(self.SnakeDirection == 1 or self.SnakeDirection == 2):
                self.SnakeDirection = 4
        
        #New position based on direction
        if (self.SnakeDirection == 1):
            new = [self.SnakeList[0][0], self.SnakeList[0][1] - SNAKE_SPEED]
        if (self.SnakeDirection == 2):
            new = [self.SnakeList[0][0], self.SnakeList[0][1] + SNAKE_SPEED]
        if (self.SnakeDirection == 3):
            new = [self.SnakeList[0][0] - SNAKE_SPEED, self.SnakeList[0][1]]
        if (self.SnakeDirection == 4):
            new = [self.SnakeList[0][0] + SNAKE_SPEED, self.SnakeList[0][1]]  

        #Moving the snake        
        for i in range(0, len(self.SnakeList)-1):
            old = self.SnakeList[i]
            self.SnakeList[i] = new
            new = old
            old = self.SnakeList[i+1]
            self.SnakeList[i+1] = old
        self.SnakeList[len(self.SnakeList)-1] = new
        
        #Collisions
        
        #Check for a screen collision
        if(self.SnakeList[0][0] < 0 or self.SnakeList[0][0] + SNAKE_WIDTH > WINDOW_WIDTH or self.SnakeList[0][1] < 0 or self.SnakeList[0][1] + SNAKE_HEIGHT > WINDOW_HEIGHT):
            return (-1)  #GAME OVER
        
        #Check for a body collision        
        for i in range(1, len(self.SnakeList) - 1):
            if(self.SnakeList[0][0] + SNAKE_WIDTH > self.SnakeList[i][0] and self.SnakeList[0][0] < self.SnakeList[i][0] + SNAKE_WIDTH and self.SnakeList[0][1] + SNAKE_HEIGHT > self.SnakeList[i][1] and self.SnakeList[0][1] < self.SnakeList[i][1] + SNAKE_HEIGHT):
                return (-1)  #GAME OVER

        #Check for a food collision
        if(self.SnakeList[0][0] + SNAKE_WIDTH > self.FoodPosition[0] and self.SnakeList[0][0] < self.FoodPosition[0] + FOOD_SIZE and self.SnakeList[0][1] + SNAKE_HEIGHT > self.FoodPosition[1] and self.SnakeList[0][1] < self.FoodPosition[1] + FOOD_SIZE):
            self.Score = self.Score + 1
            self.FoodPosition = [-100,-100]
            
            for i in range(FOOD_INCREASE_LENGTH):
                x = self.SnakeList[len(self.SnakeList) - 1][0]
                y = self.SnakeList[len(self.SnakeList) - 1][1]
                if (self.SnakeDirection == 1):
                    self.SnakeList.append([x, y + ((i+1) * SNAKE_HEIGHT)])
                if (self.SnakeDirection == 2):
                    self.SnakeList.append([x, y - ((i+1) * SNAKE_HEIGHT)])
                if (self.SnakeDirection == 3):
                    self.SnakeList.append([x + ((i+1) * SNAKE_WIDTH), y])
                if (self.SnakeDirection == 4):
                    self.SnakeList.append([x - ((i+1) * SNAKE_WIDTH), y])
            
    def draw(self):
        #Food
        food = pygame.Rect(self.FoodPosition[0], self.FoodPosition[1], FOOD_SIZE, FOOD_SIZE)
        pygame.draw.rect(self.Screen, WHITE, food)
        
        #Snake
        for i in range(len(self.SnakeList)):
            part = pygame.Rect(self.SnakeList[i][0], self.SnakeList[i][1], SNAKE_WIDTH, SNAKE_HEIGHT)
            pygame.draw.rect(self.Screen, WHITE, part)
        
    def drawInfo(self, info, action):    
        #Score
        font = pygame.font.Font(None, 28)    
        scorelabel = font.render("Score " + str(self.Score), 1, WHITE)
        self.Screen.blit(scorelabel, (30 , 10))
        
        #Information
        font = pygame.font.Font(None, 15)        
        label = font.render("step " + str(info[0]) + " ["+str(info[3])+"]", 1, WHITE)
        self.Screen.blit(label, (30 , 30))
        label = font.render("epsilon " + str(info[2]), 1, WHITE)
        self.Screen.blit(label, (30 , 45))
        label = font.render("q_max " + str(info[1]), 1, WHITE)
        self.Screen.blit(label, (30 , 60))
        actionText = "--"
        if (action[1] == 1):
            actionText = "Up"
        if (action[2] == 1):
            actionText = "Down"
        if (action[3] == 1):
            actionText = "Left"
        if (action[4] == 1):
            actionText = "Right"
        label = font.render("action " + actionText, 1, WHITE)
        self.Screen.blit(label, (30 , 75)) 
        
    def GetPresentFrame(self):        
        pygame.event.pump()  #for each frame, calls the event queue, like if the main window needs to be repainted        
        self.Screen.fill(BLACK)  #make the background black        
        self.draw()  #draw the game Screen     
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())  #copies the pixels from our surface to a 3D array. we'll use this for RL        
        pygame.display.flip()  #updates the window        
        
        return image_data  #return our surface data

    #update our Screen
    def GetNextFrame(self, action, info):
        pygame.event.pump()
        reward = self.Score  #store the current score
        self.Screen.fill(BLACK)        
        if(self.update(action) == -1):  #update the game Screen and check for game over
            reward = -1  #penalize game over
            self.newGame()
        else:
            reward = self.Score - reward  #calculate the reward based on the new score
        self.draw()  #draw the game Screen
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())  #get the surface data
        self.drawInfo(info, action)  #draw the extra information
        pygame.display.flip()  #update the window
        
        return [reward, image_data]  #return the reward and the surface data