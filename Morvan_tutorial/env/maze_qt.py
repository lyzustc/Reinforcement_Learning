import sys
import random
import time
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QDesktopWidget
from PyQt5.QtGui import QPainter
from PyQt5.QtCore import Qt, QRect, QTimer, QEventLoop

UNIT = 80
MAZE_H = 4
MAZE_W = 4

class Maze_visual(QWidget):
    ACTIONS = ['u', 'd', 'l', 'r']

    def __init__(self):
        super().__init__()
        self.setWindowTitle('maze')
        self.resize(MAZE_H * UNIT, MAZE_W * UNIT)
        self._center()
        self._build_maze()
    
    def get_actions(self):
        return self.ACTIONS

    def _state_code(self, position):
        x, y = position
        return int(y / UNIT) * MAZE_W + int(x / UNIT)

    def _center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def _build_maze(self):
        origin = np.array([UNIT/2, UNIT/2])

        hell_centers = []
        hell_centers.append(origin + np.array([UNIT * 2, UNIT]))
        hell_centers.append(origin + np.array([UNIT, UNIT * 2]))
        self.hells = []
        for center in hell_centers:
            self.hells.append(QRect(center[0] - (UNIT - 10) / 2, center[1] - (UNIT - 10) / 2,
                                    UNIT - 10, UNIT - 10))

        heaven_centers = []
        heaven_centers.append(origin + UNIT * 2)
        self.heavens = []
        for center in heaven_centers:
            self.heavens.append(QRect(center[0] - (UNIT - 10) / 2, center[1] - (UNIT - 10) / 2,
                                    UNIT - 10, UNIT - 10))

    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)

        qp.setPen(Qt.black)

        for c in range(UNIT, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            qp.drawLine(x0, y0, x1, y1)
        for r in range(UNIT, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            qp.drawLine(x0, y0, x1, y1)

        qp.setBrush(Qt.black)
        for hell in self.hells:
            qp.drawRect(hell)
        
        qp.setBrush(Qt.yellow)
        for heaven in self.heavens:
            qp.drawEllipse(heaven)

        qp.setBrush(Qt.red)
        qp.drawRect(self.agent)

        qp.end()
        
    def reset(self):
        origin = np.array([UNIT/2, UNIT/2])
        self.state = [origin[0] - (UNIT - 10) / 2, origin[1] - (UNIT - 10) / 2]
        self.agent = QRect(self.state[0], self.state[1], UNIT - 10, UNIT - 10)
        self.update()
        self.show()

        return self._state_code(self.state)

    def step(self, action):
        s = self.state
        base_action = [0, 0]
        if action == 'u':
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 'd':
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 'l':
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 'r':
            if s[0] > UNIT:
                base_action[0] -= UNIT

        s_ = [s[0]+base_action[0], s[1]+base_action[1]]
        self.state = s_

        reward = 0
        for heaven in self.heavens:
            if s_[0] == heaven.x() and s_[1] == heaven.y():
                reward = 1
                return 'terminal', reward

        for hell in self.hells:
            if s_[0] == hell.x() and s_[1] == hell.y():
                reward = -1
                return 'terminal', reward

        return self._state_code(s_), reward
  
    def render(self):
        loop = QEventLoop()
        QTimer.singleShot(200, loop.quit)
        loop.exec_()

        self.agent.moveTo(self.state[0], self.state[1])
        self.update()
        self.show()



class Maze():
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.maze = Maze_visual()

    def get_actions(self):
        return self.maze.get_actions()

    def reset(self):
        return self.maze.reset()

    def render(self):
        self.maze.render()

    def step(self, action):
        return self.maze.step(action)

    def __del__(self):
        self.render()
        sys.exit(self.app.exec_())



if __name__ == '__main__':
    maze = Maze()
    maze.reset()
    actions = maze.get_actions()

    for _ in range(100):
        i = np.random.randint(0, len(actions))
        s_, r = maze.step(actions[i])
        print("the reward is {}, new state is {}".format(r, s_))
        maze.render()