import pygame
from pygame.locals import *
import numpy as np
from test import BACKEND as bg
from datagenerator import train
import os, pyautogui, sys, cv2

camera = cv2.VideoCapture(0)
pygame.init()
pygame.display.set_caption('Gesture Control')

screen_length = 800
screen_width = 600
screen = pygame.display.set_mode((screen_length, screen_width))

#Colors
BLACK  = (  0, 0,  0)
WHITE  = ( 255, 255, 255)
RED  = ( 255,  0,  0)
BLUE  = (  0,  0, 255)
PURPLE   = ( 255,   0, 255)
color_light = (170,170,170)
color_dark = (100,100,100)

textFont = pygame.font.SysFont('Calibri', 25, True, False)
headingFont =  pygame.font.SysFont('Calibri', 50, True, False)

directory = 'data/train/'

trigger, prev_pred, click = False, False, False
start, Calibration, Selector, Test, Train, loadDisplay = True, False, False, False, False, False


def callback(value):
   pass


try:
    while True:
        ret, frame = camera.read()

        if start == True:

            screen.fill(BLACK)
            heading = headingFont.render("G E S T U R E  C O N T R O L", True, WHITE)
            screen.blit(heading, [140, 200])

            subheading = headingFont.render("O P E N C V", True, RED)
            screen.blit(subheading, [275, 250])

            text = textFont.render("Press 'ANY KEY' to continue", True, WHITE)
            screen.blit(text, [255, 470])
            pygame.display.update()
            #pygame.display.update()

            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    start = False
                    Calibration = True

        elif Calibration == True:
            mouse = pygame.mouse.get_pos()
            screen.fill(BLACK)

            heading = textFont.render("C A L I B R A T I O N", True, WHITE)
            screen.blit(heading, [300, 25])

            width, height = 300, 40
            x, y = 450, 530
            cx, cy = 540, 540

            hsv = bg.calibrate(frame)


            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (370, 230))
            frame = cv2.flip(frame, 1)
            frame = frame.swapaxes(0,1)
            frame = pygame.surfarray.make_surface(frame)

            thresh = cv2.cvtColor(hsv, cv2.COLOR_BGR2RGB)
            thresh = cv2.resize(thresh, (370, 230))
            thresh = cv2.flip(thresh, 1)
            thresh = thresh.swapaxes(0,1)
            thresh = pygame.surfarray.make_surface(thresh)

            text = textFont.render("(Original)", True, WHITE)
            screen.blit(text, [550, 190])
            text = textFont.render("(Thresh)", True, WHITE)
            screen.blit(text, [575, 450])

            pygame.draw.rect(screen,WHITE,[x, y, width, height])
            text = textFont.render("Auto Calibrate", True, BLACK)
            screen.blit(text, [cx , cy])

            screen.blit(frame, (30,75))
            screen.blit(thresh, (30,335))
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        Calibration = False
                        Selector = True
                        auto = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if x <= mouse[0] <= x + width and y <= mouse[1] <= y + height:
                        Calibration = False
                        Selector = True
                        auto = True

        elif Selector == True:
            mouse = pygame.mouse.get_pos()
            screen.fill(BLACK)
            train_count = False

            width, height = 240, 40
            x, y = 280, 150
            cx, cy = x + width//4 + 20, y + height//4

            pygame.draw.rect(screen,WHITE,[x, y, width, height])
            pygame.draw.rect(screen,WHITE,[x, y + 100, width, height])
            pygame.draw.rect(screen,WHITE,[x, y + 200, width, height])

            text = textFont.render("Load", True, BLACK)
            screen.blit(text, [cx, cy])
            text = textFont.render("Train", True, BLACK)
            screen.blit(text, [cx, cy + 100])
            text = textFont.render("Recalibrate", True, BLACK)
            screen.blit(text, [cx - 20, cy + 200])

            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if x <= mouse[0] <= x + width and y <= mouse[1] <= y + height:
                        Test = True
                        Selector = False
                    elif x <= mouse[0] <= x + width and y + 100 <= mouse[1] <= y + 100 + height:
                        Train = True
                        Selector = False
                    elif x <= mouse[0] <= x + width and y + 200 <= mouse[1] <= y + 200 + height:
                        Calibration = True
                        Selector = False

        elif Test == True:
            mouse = pygame.mouse.get_pos()
            screen.fill(BLACK)

            width, height = 200, 40
            x, y = 40, 250
            cx, cy = x + width//4 + 10, y + height//4

            pygame.draw.rect(screen,WHITE,[x, y, width, height])
            pygame.draw.rect(screen,WHITE,[x + 260, y, width, height])
            pygame.draw.rect(screen,WHITE,[x + 515, y, width, height])

            text = textFont.render("Static", True, BLACK)
            screen.blit(text, [cx, cy])
            text = textFont.render("Dynamic", True, BLACK)
            screen.blit(text, [cx + 260, cy])
            text = textFont.render("Cursor", True, BLACK)
            screen.blit(text, [cx + 515, cy])

            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if x <= mouse[0] <= x + width and y <= mouse[1] <= y + height:
                        mode = "Static"
                        loadDisplay = True
                        Test = False
                    elif x + 260 <= mouse[0] <= x + 260 + width and y <= mouse[1] <= y + height:
                        loadDisplay = True
                        mode = "Dynamic"
                        loadDisplay = True
                        Test = False
                    elif x + 515 <= mouse[0] <= x + 515 + width and y <= mouse[1] <= y + height:
                        loadDisplay = True
                        mode = "Cursor"
                        loadDisplay = True
                        Test = False

        elif Train == True:
            mouse = pygame.mouse.get_pos()
            screen.fill(BLACK)

            width, height = 300, 40
            x, y = 450, 530
            cx, cy = x + width//4 + 50, y + height//4

            # try:
            count, roi = train.main(frame)
            roi_arr = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi_arr = cv2.resize(roi_arr, (370, 230))
            roi_arr = cv2.flip(roi_arr, 1)
            roi_arr = roi_arr.swapaxes(0,1)
            roi_arr = pygame.surfarray.make_surface(roi_arr)

            screen.blit(roi_arr, (30,330))

            text = textFont.render("DATA COUNT" , True, WHITE)
            screen.blit(text, [465, 60])

            for i in range(len(count)):
               text = textFont.render(str(i) + ":      " + str(count[i]) , True, WHITE)
               screen.blit(text, [525, 120 + 50*i])

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (370, 230))
            frame = cv2.flip(frame, 1)
            frame = frame.swapaxes(0,1)
            frame = pygame.surfarray.make_surface(frame)

            #FEED
            text = textFont.render("Live Feed", True, WHITE)
            screen.blit(text, [30, 25])
            text = textFont.render("ROI", True, WHITE)
            screen.blit(text, [30, 300])

            #RETURN BUTTON
            pygame.draw.rect(screen,WHITE,[x, y, width, height])
            text = textFont.render("Back", True, BLACK)
            screen.blit(text, [cx, cy])

            screen.blit(frame, (30,50))

            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if x <= mouse[0] <= x + width and y <= mouse[1] <= y + height:
                        Train = False
                        Selector = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_0:
                        try:
                            cv2.imwrite(directory+'0/'+str(count[0])+'.jpg', roi)
                        except:
                            os.makedirs(directory+'0/')
                    if event.key == pygame.K_1:
                        try:
                            cv2.imwrite(directory+'1/'+str(count[1])+'.jpg', roi)
                        except:
                            os.makedirs(directory+'1/')
                    if event.key == pygame.K_2:
                        try:
                            cv2.imwrite(directory+'2/'+str(count[2])+'.jpg', roi)
                        except:
                            os.makedirs(directory+'2/')
                    if event.key == pygame.K_3:
                        try:
                            cv2.imwrite(directory+'3/'+str(count[3])+'.jpg', roi)
                        except:
                            os.makedirs(directory+'3/')
                    if event.key == pygame.K_4:
                        try:
                            cv2.imwrite(directory+'4/'+str(count[4])+'.jpg', roi)
                        except:
                            os.makedirs(directory+'4/')
                    if event.key == pygame.K_5:
                        try:
                            cv2.imwrite(directory+'5/'+str(count[5])+'.jpg', roi)
                        except:
                            os.makedirs(directory+'5/')
                    if event.key == pygame.K_6:
                        try:
                            cv2.imwrite(directory+'6/'+str(count[6])+'.jpg', roi)
                        except:
                            os.makedirs(directory+'6/')
                    if event.key == pygame.K_7:
                        try:
                            cv2.imwrite(directory+'7/'+str(count[7])+'.jpg', roi)
                        except:
                            os.makedirs(directory+'7/')
                    if event.key == pygame.K_8:
                        try:
                            cv2.imwrite(directory+'8'+str(count[8])+'.jpg', roi)
                        except:
                            os.makedirs(directory+'8/')
                    if event.key == pygame.K_9:
                        try:
                            cv2.imwrite(directory+'9'+str(count[8])+'.jpg', roi)
                        except:
                            os.makedirs(directory+'9/')

            pygame.display.update()


        elif loadDisplay == True:
            pred = None
            mouse = pygame.mouse.get_pos()
            screen.fill(BLACK)

            width, height = 300, 40
            x, y = 450, 530
            cx, cy = x + width//4 + 50, y + height//4

            try:
                roi, pred, mx, my = bg.predict(frame, auto)
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                roi = cv2.resize(roi, (370, 230))
                roi = cv2.flip(roi, 1)
                roi = roi.swapaxes(0,1)
                roi = pygame.surfarray.make_surface(roi)

                text = textFont.render(str(pred), True, WHITE)
                screen.blit(text, [550, 60])

                screen.blit(roi, (30,330))
            except:
                pygame.draw.rect(screen,WHITE,[30, 330, 370, 230])

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (370, 230))
            frame = cv2.flip(frame, 1)
            frame = frame.swapaxes(0,1)
            frame = pygame.surfarray.make_surface(frame)

            #FEED
            text = textFont.render("Live Feed", True, WHITE)
            screen.blit(text, [30, 25])
            text = textFont.render("ROI", True, WHITE)
            screen.blit(text, [30, 300])

            #GESTURE
            text = textFont.render("Gesture:", True, WHITE)
            screen.blit(text, [450, 60])
            #MODE
            text = textFont.render("Mode:", True, WHITE)
            screen.blit(text, [450, 115])
            text = textFont.render(mode, True, WHITE)
            screen.blit(text, [530, 115])

            #RETURN BUTTON
            pygame.draw.rect(screen,WHITE,[x, y, width, height])
            text = textFont.render("Back", True, BLACK)
            screen.blit(text, [cx, cy])

            screen.blit(frame, (30,50))
            pygame.display.update()

            if mode == "Cursor":
                try:
                    if pred == 0:
                        pyautogui.moveTo(1920 - mx, my)
                        click = True
                    elif pred == 3 and click == True:
                        pyautogui.click()
                        click = False
                except:
                    pass
            elif mode == "Dynamic":
                try:
                    if pred == 3:
                        trigger = True
                    elif pred == 0 and trigger == True:
                        pyautogui.press('volumemute')
                        trigger = False
                    else:
                        trigger = False
                except:
                    pass

            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if x <= mouse[0] <= x + width and y <= mouse[1] <= y + height:
                        loadDisplay = False
                        Selector = True


except (KeyboardInterrupt,SystemExit):
    pygame.quit()
    cv2.destroyAllWindows()
